
import torch

from .superparameters_search import ray_tune_main, set_superparameters
from .utils import simple_total_variation, BNForwardLossHook
from .utils import compute_label_acc
from .superparameters_search import HSGradientInversion
from .MLCO import MLCOGradientInversion
from .gradient_inversion import BaseGradientInversion

__all__ = [
    'ray_tune_main',
    'set_superparameters',
    'simple_total_variation',
    'BNForwardLossHook',
    'init_x_pseudo',
    'get_y_pseudo',
    'reconstruction'
]


def init_x_pseudo(args):

    '''return x_fake.shape = (n_seed, batch_size, n_channel, W, H)'''

    if not args.superparameters_search:
        x_pseudo_list = torch.randn(args.n_seed, *args.input_size)
    else:
        x_pseudo_list = torch.randn(*args.input_size)

    print(f'initialize list of x_pseudo: {x_pseudo_list.shape}')

    return x_pseudo_list

def dgi_label_recon(grads, b, N):

    '''
    b: batch size
    N: n_classes
    '''
    g = grads[-2].sum(-1)
    C = g[torch.where(g > 0)[0]].max()
    m = N * C / b

    pred_label = []
    for i, gi in enumerate(g):
        if gi < 0:
            pred_label.append(i)
            g[i] += m
    while len(pred_label) < b:
        idx = g.argmin().item()
        pred_label.append(idx)
        g[idx] += m

    return torch.as_tensor(pred_label)

def get_y_pseudo(args, target_gradient, metric_dict):

    if args.pseudo_label_init == 'known':
        label_pred = metric_dict['y_true']
    elif args.pseudo_label_init == 'from_grads':
        label_pred = dgi_label_recon(target_gradient, args.batch_size, args.n_classes).detach().view(-1,1)
        print(f'[dgi info] y pred: {label_pred.view(-1,).cpu().numpy()}')

    n_correct, acc = compute_label_acc(metric_dict['y_true'], label_pred)
    print(f' > label acc: {acc * 100:.2f}%')

    return label_pred.view(-1,)


def reconstruction(
        x_pseudo_list,
        y_pseudo,
        target_gradient,
        model,
        dm,
        ds,
        args,
        metric_dict=None,
        bn_loss_layers=None):

    lr_start_end = (args.lr, 1e-3)
    regularization = set_regularization(args)

    if args.superparameters_search:
        gi = HSGradientInversion(
                target_gradient,
                model,
                dm,
                ds,
                args)

        x_pseudo, loss_track = gi.run(
            lr_start_end,
            regularization,
            x_pseudo_list,
            y_pseudo,
            args.epochs,
            metric_dict=metric_dict,
            bn_loss_layers=bn_loss_layers)

    elif args.MinCombine:
        gi = MLCOGradientInversion(
                target_gradient, model,
                dm, ds, args)
        x_pseudo, loss_track = gi.multigroup_run(
            lr_start_end, regularization,
            x_pseudo_list, y_pseudo,
            bn_loss_layers=bn_loss_layers
        )

    else:
        gi = BaseGradientInversion(
                target_gradient, model,
                dm, ds, args)

        new_x_pseudo_list, loss_track_list = gi.multigroup_run(
            lr_start_end, regularization,
            x_pseudo_list, y_pseudo,
            args.epochs, bn_loss_layers=bn_loss_layers)

        # choose the result
        result_losses = torch.ones(size=(args.n_seed,))
        for i, loss_track in enumerate(loss_track_list):
            result_losses[i] = loss_track['gradient loss']
        index_best = torch.argmin(result_losses)
        x_pseudo = new_x_pseudo_list[index_best]
        print(f'choose the best results {index_best}.')
        print_str = '[end results] > '
        for key, value in loss_track_list[index_best].items():
            print_str += f'[{key}]: {value:.4f} '
        print(print_str)

    return x_pseudo, loss_track


def set_regularization(args):

    regularization = dict()
    if args.TV > 0:
        regularization['TV'] = args.TV
    if args.BN > 0:
        regularization['BN'] = args.BN
    if args.input_norm != 0:
        regularization['input_norm'] = args.input_norm

    return regularization
