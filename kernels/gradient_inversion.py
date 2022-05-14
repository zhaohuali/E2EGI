
import torch
import torch.nn as nn
import torch.distributed as dist

from .utils import simple_total_variation


class BaseGradientInversion():

    def __init__(
            self, target_gradient,
            model, dm, ds, args) -> None:

        self.target_gradient = target_gradient
        self.gpu = args.gpu
        self.gradient_loss_fun = args.gradient_loss_fun
        self.grads_weight = args.grads_weight
        self.model = model
        self.input_boxed = args.input_boxed
        self.grad_sign = args.grad_sign
        self.dm = dm
        self.ds = ds
        self.input_noise = args.input_noise
        self.print_freq = args.print_freq
        self.optim = args.optim
        self.min_grads_loss = args.min_grads_loss
        self.distributed = args.distributed
        self.ngpus_per_node = args.ngpus_per_node

        if args.model_eval:
            self.model.eval()
        else:
            self.model.train()

        self.criterion = nn.CrossEntropyLoss()

    def setup(
            self, regularization,
            bn_loss_layers, x_group):

        self.loss_track = dict()
        self.loss_track['gradient loss'] = float('inf')
        self.loss_track['total loss'] = float('inf')
        self.best_x_pseudo = None
        self.best_loss_track = dict()
        self.best_loss_track['gradient loss'] = float('inf')

        self.regularization = regularization
        self.bn_loss_layers = bn_loss_layers
        self.x_group = x_group

    def multigroup_run(
            self, lr_start_end,
            regularization, x_pseudo_list,
            y_pseudo, epochs,
            bn_loss_layers=None, x_group=None,
            state='update'):

        n_seed = len(x_pseudo_list)
        new_x_pseudo_list = []
        loss_track_list = []

        for i_seed in range(n_seed):
            print(f'[restart] {i_seed+1} / {n_seed}')

            x_pseudo, loss_track = self.run(
                lr_start_end,
                regularization, x_pseudo_list[i_seed],
                y_pseudo, epochs,
                bn_loss_layers=bn_loss_layers, x_group=x_group,
                state=state)

            new_x_pseudo_list.append(x_pseudo)
            loss_track_list.append(loss_track)

        return new_x_pseudo_list, loss_track_list

    def run(self, lr_start_end: tuple,
            regularization: dict, x_pseudo,
            y_pseudo, epochs: int,
            bn_loss_layers=None, x_group=None,
            state='update', flag='base'):
        '''
        lr_start_end: tuple (lr_start, lr_end)
        regularization: dict ('TV': value, .., )
        '''
        self.setup(regularization, bn_loss_layers, x_group)

        x_pseudo = x_pseudo.detach().clone()
        x_pseudo.requires_grad = True

        if self.optim == 'Adam':
            optimizer = torch.optim.Adam([x_pseudo], lr_start_end[0])

        scheduler = self.get_lr_scheduler(optimizer, epochs, lr_start_end[1])

        try:
            for epoch in range(1, epochs+1):
                closure = self.grads_inversion(x_pseudo, y_pseudo,
                                               optimizer, state)
                optimizer.step(closure)
                scheduler.step()

                with torch.no_grad():

                    # add noise to x_pseudo
                    if self.input_noise > 0:
                        x_pseudo.data = self.add_noise(x_pseudo, optimizer)

                    # boxing x_pseudo
                    if self.input_boxed:
                        x_pseudo.data = self.boxing_input(x_pseudo)

                    if (epoch == 1
                            or epoch % self.print_freq == 0
                            or epoch == epochs):
                        self.display_results(epoch, epochs, flag)

        except KeyboardInterrupt:
            print('early stop.')
            self.display_results(epoch, epochs, flag)

        if self.min_grads_loss:
            return self.best_x_pseudo, self.best_loss_track
        else:
            return x_pseudo, self.loss_track

    def get_optimizer(self):
        pass

    def get_lr_scheduler(self, optimizer, epochs, lr_end):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs+10,
            T_mult=1,
            eta_min=lr_end)
        return scheduler

    def grads_inversion(
            self, x_pseudo, y_pseudo,
            optimizer=None, state='update'):

        def closure():

            self.model.zero_grad()

            if optimizer is not None:
                optimizer.zero_grad()

            out_pseudo = self.model(x_pseudo)
            loss_pseudo = self.criterion(out_pseudo, y_pseudo)

            if (state == 'update'):
                create_graph = True
            elif (state == 'get_gradient_loss'
                    or state == 'get_total_loss'):
                create_graph = False
            else:
                raise ValueError(f'state:{state}')

            gradent_pseudo = torch.autograd.grad(
                loss_pseudo,
                filter(lambda p: p.requires_grad, self.model.parameters()),
                create_graph=create_graph)

            if self.distributed:
                self.average_gradients(gradent_pseudo, self.ngpus_per_node)

            gradient_loss = self.compute_loss(
                x_pseudo, gradent_pseudo, state=state)

            self.update_best_x_pseudo(x_pseudo)

            if (state == 'update'):
                gradient_loss.backward()
                if self.grad_sign:
                    x_pseudo.grad.sign_()

            return gradient_loss

        return closure

    def compute_loss(
            self, x_pseudo,
            gradent_pseudo,
            state='update'):

        loss = self.compute_grads_loss(gradent_pseudo)
        self.loss_track['gradient loss'] = loss.item()

        if state == 'get_gradient_loss':
            return loss

        loss += self.compute_regularization(x_pseudo)
        self.loss_track['total loss'] = loss

        return loss

    def compute_grads_loss(self, gradent_pseudo):

        loss = 0
        grads_indices = range(len(self.target_gradient))

        if self.gradient_loss_fun == 'sim':
            sim_diff = 0
            pnorm = [0, 0]
            for i in grads_indices:
                sim_diff -= (self.target_gradient[i] * gradent_pseudo[i]).sum()
                pnorm[0] += gradent_pseudo[i].pow(2).sum()
                pnorm[1] += self.target_gradient[i].pow(2).sum()
            sim_diff = 1 + sim_diff / pnorm[0].sqrt() / pnorm[1].sqrt()
            loss += sim_diff * self.grads_weight

        elif self.gradient_loss_fun == 'L2':
            L2_diff = 0
            for i in grads_indices:
                _diff = ((self.target_gradient[i]-gradent_pseudo[i])**2).sum()
                L2_diff += _diff
            grads_diff = L2_diff
            loss += grads_diff * self.grads_weight

        return loss

    def compute_regularization(self, x_pseudo):

        loss = 0

        TV_diff = simple_total_variation(x_pseudo)
        x_norm_loss = torch.norm(
            x_pseudo.view(x_pseudo.size(0), -1), p=2, dim=-1).mean()
        self.loss_track['TV loss'] = TV_diff.item()
        self.loss_track['input norm'] = x_norm_loss

        if ('TV' in self.regularization
                and self.regularization['TV'] > 0):
            loss += self.regularization['TV'] * TV_diff

        if ('BN' in self.regularization
                and self.regularization['BN'] > 0):
            bn_loss = sum([mod.bn_loss for mod in self.bn_loss_layers])
            loss += self.regularization['BN'] * bn_loss
            self.loss_track['BN loss'] = bn_loss

        if ('Group' in self.regularization
                and self.regularization['Group'] > 0):
            group_loss = ((x_pseudo - self.x_group) ** 2).mean()
            loss += self.regularization['Group'] * group_loss
            self.loss_track['Group loss'] = group_loss.item()

        if ('input_norm' in self.regularization
                and self.regularization['input_norm'] != 0):
            loss += self.regularization['input_norm'] * x_norm_loss

        return loss

    def update_best_x_pseudo(self, x_pseudo):

        with torch.no_grad():
            if (self.loss_track['gradient loss'] <
                    self.best_loss_track['gradient loss']):
                self.best_x_pseudo = x_pseudo.detach().clone()
                if self.input_boxed:
                    self.best_x_pseudo.data = \
                        self.boxing_input(self.best_x_pseudo)

                self.best_loss_track['gradient loss'] = \
                    self.loss_track['gradient loss']
                self.best_loss_track['total loss'] = \
                    self.loss_track['total loss']
                for key, value in self.loss_track.items():
                    self.best_loss_track[key] = value

    def boxing_input(self, x_pseudo):

        return torch.max(torch.min(x_pseudo,
                                   (1 - self.dm) / self.ds),
                         -self.dm / self.ds)

    def add_noise(self, x_pseudo, optimizer):

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        randn_x = now_lr * self.input_noise * torch.randn_like(x_pseudo)
        return x_pseudo + randn_x

    def display_results(self, epoch, epochs, flag=''):

        print(f'>> [{epoch}/{epochs}]')

        best_str = f'{flag}-best > '
        for key, value in self.best_loss_track.items():
            best_str += f'[{key}] {value:.4f} '
        print(best_str)

        now_str = f'{flag}-now  > '
        for key, value in self.loss_track.items():
            now_str += f'[{key}] {value:.4f} '
        print(now_str)

    def average_gradients(self, gradent_pseudo, ngpus_per_node):

        size = float(ngpus_per_node)
        for grad in gradent_pseudo:
            dist.all_reduce(grad.data, op=dist.ReduceOp.SUM)
            grad.data /= size
