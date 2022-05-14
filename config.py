
import os
import sys
import logging

import apex
import torch
import torchvision.models as models

from kernels import simple_total_variation, BNForwardLossHook


def save_args(args):

    config_bk = dict(
        world_size=args.world_size,
        rank=args.rank,
        multiprocessing_distributed=args.multiprocessing_distributed,
        epochs=args.epochs,
        print_freq=args.print_freq,
        GInfoR=args.GInfoR)
    args.world_size = -1
    args.rank = -1
    args.multiprocessing_distributed = False
    args.epochs = args.HP_epochs
    # Gradient inversion results are not printed in hyperparameter search
    args.print_freq = sys.maxsize
    args.GInfoR = False

    return config_bk


def update_args(args, HP_config, config_bk):

    if config_bk is not None:
        for key, value in config_bk.items():
            args.__dict__[key] = value

    for key, value in HP_config.items():
        args.__dict__[key] = value

    args.superparameters_search = False


def build_history(args, filename='results/exp'):

    if args.superparameters_search:
        history_path = os.getcwd()
    else:
        filename = os.path.join(filename, args.id)
        history_path = os.path.join(args.root, filename)

    if (not args.multiprocessing_distributed
            or (args.multiprocessing_distributed
                and args.rank % args.ngpus_per_node == 0)):
        if not os.path.exists(history_path):
            os.mkdir(history_path)
        filename = f'{args.id}-{args.clock}.log'
        filepath = os.path.join(history_path, filename)
        sys.stdout = Logger(args.id, filepath, sys.stdout)
        print(f'create file: {filepath}')

    args.path = history_path


class Logger(object):
    def __init__(self, id, filename='default.log', stream=sys.stdout):

        self.terminal = stream

        logging.basicConfig(
            filename=filename,
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(id)
        self.log = logger

    def write(self, message):
        self.terminal.write(message)
        if message != '\n':
            self.log.warning(message)
        if 'finish' in message:
            self.log.critical(message)

    def flush(self):
        pass


def load_checkpoint(args):

    if args.superparameters_search:
        checkpoint = torch.load(args.simulation_checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

    target_clock = checkpoint['clock']
    targer_args = checkpoint['args']

    # input
    batch_size = checkpoint['batch_size']
    dataset_setup = checkpoint['dataset_setup']
    mean = dataset_setup['mean']
    std = dataset_setup['std']
    n_channels, W, H = dataset_setup['size']
    n_classes = dataset_setup['n_classes']
    input_size = (batch_size, n_channels, W, H)
    dm = torch.as_tensor(mean).view(n_channels, 1, 1)
    ds = torch.as_tensor(std).view(n_channels, 1, 1)

    # model
    arch = checkpoint['arch']
    model_eval = checkpoint['model_eval']
    model_state_dict = checkpoint['state_dict']
    if 'free_last_layers_list' in checkpoint:
        free_last_layers_list = checkpoint['free_last_layers_list']
    else:
        free_last_layers_list = ['all']
    model = load_model(
        arch, model_state_dict, free_last_layers_list, n_classes)

    # target
    target_gradient = checkpoint['target_gradient']
    bn_mean_list = checkpoint['bn_mean_list']
    bn_var_list = checkpoint['bn_var_list']

    # test
    metric_dict = dict()
    if args.metric:
        metric_dict['x_true'] = checkpoint['x_true']
        metric_dict['y_true'] = checkpoint['y_true']
        get_target_state(metric_dict['x_true'])

    # record
    args.batch_size = batch_size
    args.n_classes = n_classes
    args.input_size = input_size
    args.arch = arch
    args.model_eval = model_eval

    if not args.distributed or args.gpu == 0:

        print(f'\n========== Target: {target_clock} ========')
        for arg in vars(targer_args):
            msg = f'{arg:30} {getattr(targer_args, arg)}'
            print(msg)

        print(f'\n========== GI: {args.clock} ========')
        for arg in vars(args):
            msg = f'{arg:30} {getattr(args, arg)}'
            print(msg)
        full_norm = torch.stack([g.norm() for g in target_gradient]).mean()
        print(f'[Num of grads] {len(target_gradient)}')
        print(f'[Full gradient norm is] {full_norm:e}.')

    return model, bn_mean_list, bn_var_list, \
        dm, ds, target_gradient, metric_dict


def load_model(arch, state_dict, free_last_layers_list, n_classes):

    model = models.__dict__[arch](pretrained=False)

    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]

    if arch.startswith('resnet') and n_classes != 1000:
        model.fc = torch.nn.Linear(in_features=model.fc.in_features,
                                   out_features=n_classes)

    model.load_state_dict(state_dict, strict=True)

    if 'all' not in free_last_layers_list:
        for name, param in model.named_parameters():
            for freed_name in free_last_layers_list:
                if freed_name in name:
                    param.requires_grad = True
                    print(f'[freed layers] {name}')
                    break
                else:
                    param.requires_grad = False
    else:
        print('[freed layers] all')

    return model


def get_target_state(x_true):

    TV_diff = simple_total_variation(x_true)
    x_norm_loss = \
        torch.norm(x_true.view(x_true.size(0), -1), p=2, dim=-1).mean()
    print(f'[Target] > [TV] {TV_diff:.4f} [input norm] {x_norm_loss:.4f}')


def set_BN_regularization(bn_mean_list, bn_var_list, model, args):
    bn_loss_layers = []
    i_bn_layers = 0
    for module in model.modules():
        if isinstance(module, apex.parallel.SyncBatchNorm):
            if bn_mean_list is not None and args.exact_bn:
                mean = \
                    bn_mean_list[i_bn_layers].detach().clone().cuda(args.gpu)
                var = \
                    bn_var_list[i_bn_layers].detach().clone().cuda(args.gpu)
            else:
                mean = module.running_mean.detach().clone()
                var = module.running_var.detach().clone()
            bn_loss_layers.append(
                BNForwardLossHook(
                    module,
                    args.distributed,
                    mean,
                    var))
            if i_bn_layers == 0:
                print(f'set loss hook for \
                    bn statistics,exact_bn [{args.exact_bn}]')
            i_bn_layers += 1

    return bn_loss_layers
