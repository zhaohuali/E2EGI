r"""
MLCO: Minimum loss combinatorial optimization
"""

import math

import torch
import torch.distributed as dist

from .gradient_inversion import BaseGradientInversion
from .utils import save_imgs


class MLCOGradientInversion(BaseGradientInversion):

    def __init__(self, target_gradient, model, dm, ds, args) -> None:
        super().__init__(target_gradient, model, dm, ds, args)

        self.Group = args.Group
        self.total_epochs = args.epochs
        self.T_init = math.ceil(args.T_init_rate * args.epochs)
        self.total_T_in = \
            math.ceil(args.total_T_in_rate * args.epochs/args.T_in)
        self.T_in = args.T_in
        self.T_end = math.ceil(args.T_end_rate * args.epochs)
        self.path = args.path
        self.clock = args.clock
        self.input_size = args.input_size

    def multigroup_run(
            self, lr_start_end,
            regularization, x_pseudo_list,
            y_pseudo, bn_loss_layers=None):

        lr_dict = self.get_lr_dict(lr_start_end)
        n_seed = len(x_pseudo_list)
        rank = 0
        if self.distributed:
            rank = dist.get_rank()
        # Initial Reconstruction
        init_lr_start = lr_dict['init_lr_start']
        init_lr_end = lr_dict['init_lr_end']
        loss_list = []
        for i_seed in range(n_seed):
            x_pseudo_list[i_seed], loss_track = self.run(
                (init_lr_start, init_lr_end),
                regularization, x_pseudo_list[i_seed],
                y_pseudo, self.T_init,
                bn_loss_layers=bn_loss_layers,
                flag=f'Init-seed({i_seed+1}/{n_seed})')

            loss_list.append(loss_track['gradient loss'])
            save_imgs(
                x_pseudo_list[i_seed], y_pseudo, self.dm, self.ds,
                self.distributed, self.input_size, rank, self.path,
                self.clock, str_=f'Init-seed({i_seed+1}_{n_seed})')

        # Minimum Loss Combinatorial Optimization
        x_group = self.get_best_pseudo(loss_list, x_pseudo_list, y_pseudo)

        for i in range(self.total_T_in):
            in_lr_start = lr_dict['in_lr_start'+str(i)]
            in_lr_end = lr_dict['in_lr_end'+str(i)]
            loss_list = []
            for i_seed in range(n_seed):
                regularization['Group'] = self.Group
                x_pseudo_list[i_seed], loss_track = self.run(
                    (in_lr_start, in_lr_end),
                    regularization, x_pseudo_list[i_seed],
                    y_pseudo, self.T_in,
                    bn_loss_layers=bn_loss_layers, x_group=x_group,
                    flag=f'MLCO-in({i+1}/{self.total_T_in})' +
                    f'-seed({i_seed+1}/{n_seed})')

                loss_list.append(loss_track['gradient loss'])

            x_group = self.get_best_pseudo(loss_list, x_pseudo_list, y_pseudo)
            save_imgs(
                x_group, y_pseudo, self.dm, self.ds,
                self.distributed, self.input_size, rank, self.path,
                self.clock, str_=f'MLCO-in({i+1}_{self.total_T_in})-group')

        # Final Reconstruction
        self.input_noise = 0
        del regularization['Group']
        end_lr_start = lr_dict['end_lr_start']
        end_lr_end = lr_dict['end_lr_end']
        x_pseudo, loss_track = self.run(
            (end_lr_start, end_lr_end),
            regularization, self.x_group,
            y_pseudo, self.T_end,
            bn_loss_layers=bn_loss_layers, flag='Final')

        return x_pseudo, loss_track

    def get_best_pseudo(self, loss_list, x_pseudo_list, y_pseudo):

        # get better x_pseudo
        loss_list = torch.as_tensor(loss_list)
        min_loss_idx = loss_list.argmin()
        x_pseudo = x_pseudo_list[min_loss_idx].detach().clone()
        min_loss = self.grads_inversion(
            x_pseudo, y_pseudo, state='get_gradient_loss')().item()
        print(f'orig loss {min_loss}')

        n_seed = x_pseudo_list.shape[0]
        batch_size = x_pseudo_list.shape[1]

        rank = 0
        if self.distributed:
            rank = dist.get_rank()
            self.distributed_wait(
                x_pseudo, y_pseudo, batch_size,
                n_seed, min_loss_idx, min_loss, wait='init')

        min_loss = self.grads_inversion(
            x_pseudo, y_pseudo, state='get_gradient_loss')().item()
        for i_samples in range(batch_size):
            x_test = x_pseudo.detach().clone()
            best_seed = min_loss_idx

            for i_seed in range(n_seed):
                x_test[i_samples] = x_pseudo_list[i_seed][i_samples]
                loss = self.grads_inversion(
                    x_test, y_pseudo, state='get_gradient_loss')().item()
                if loss < min_loss:
                    min_loss = loss
                    best_seed = i_seed

            x_pseudo[i_samples] = \
                x_pseudo_list[best_seed][i_samples].detach().clone()
            k = i_samples + rank*batch_size
            print(f'i_imgs: {k} best_seed:{best_seed} loss: {min_loss}')

        if self.distributed:
            self.distributed_wait(
                x_pseudo, y_pseudo, batch_size,
                n_seed, min_loss_idx, min_loss, wait='end')

        min_loss = self.grads_inversion(
            x_pseudo, y_pseudo, state='get_gradient_loss')().item()
        print(f'min loss: {min_loss}')

        return x_pseudo.detach().clone()

    def distributed_wait(
            self, x_pseudo, y_pseudo,
            batch_size, n_seed, min_loss_idx,
            min_loss, wait='init'):

        rank = dist.get_rank()
        n_ranks = dist.get_world_size()

        if wait == 'init':
            n_wait_rank = rank
            s_rank = 1
        else:
            n_wait_rank = n_ranks - rank - 1
            s_rank = rank + 1

        for i_rank in range(n_wait_rank):
            for i_samples in range(batch_size):
                best_seed = min_loss_idx
                for i_seed in range(n_seed):
                    loss = self.grads_inversion(
                        x_pseudo, y_pseudo, state='get_gradient_loss')().item()
                    if loss < min_loss:
                        min_loss = loss
                        best_seed = i_seed
                if rank == 0:
                    k = i_samples + (i_rank+s_rank) * batch_size
                    print(
                        f'i_imgs: {k} best_seed:{best_seed} ' +
                        f'loss: {min_loss}')

    def get_lr_dict(self, lr_start_end):
        'split learning rate'
        lr_start, lr_end = lr_start_end
        assert lr_start > lr_end
        lr_interval = lr_start - lr_end
        lr_dict = dict()

        # set "init" lr
        T_total = self.T_init + self.T_in*self.total_T_in + self.T_end
        lr_dict['init_lr_start'] = lr_start
        now_lr = lr_start - lr_interval*(self.T_init/T_total)
        lr_dict['init_lr_end'] = now_lr

        # set "in" lr
        for i in range(self.total_T_in):
            lr_dict['in_lr_start'+str(i)] = now_lr
            now_lr = now_lr - lr_interval*(self.T_in/T_total)
            lr_dict['in_lr_end'+str(i)] = now_lr

        # set "end" lr
        lr_dict['end_lr_start'] = now_lr
        lr_dict['end_lr_end'] = lr_end

        print('[lr manager]')
        for key, value in lr_dict.items():
            print(f'{key}: {value}')

        return lr_dict
