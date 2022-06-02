r"""
Code for performing Gradient Inversion

Execution mode of the code (examples):

"""

import os
import random
import builtins
import argparse
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import save_args, update_args, build_history
from config import load_checkpoint, set_BN_regularization
from kernels import ray_tune_main, set_superparameters
from kernels import init_x_pseudo, get_y_pseudo, reconstruction
from metric import get_gir, save_results_with_metric
from distributed import set_distributed, get_rank_samples

__version__ = '0.1.0'

parser = argparse.ArgumentParser(description='E2EGI')
parser.add_argument('--id', default='test', type=str, help='code ID')
parser.add_argument('--root', default='./', type=str)

''' Configuration of Hardware'''
parser.add_argument('--world-size', default=-1, type=int,
                    help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='Node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='Url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='Distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch'
                         'N processes per node, which has N GPUs. This is the'
                         'fastest way to use PyTorch for either single node or'
                         'multi node data parallel training')

''' Configuration of the target environment '''
parser.add_argument('--checkpoint', type=str,
                    help='Path of checkpoint '
                         '(result of performing training.py)')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed of initializing pseudo-samples. ')
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 500)')
parser.add_argument('--exact-bn', action='store_true',
                    help='True: provide the mean and variance of the '
                         'BatchNorm computed from the target samples. If '
                         'False (default), the target statistics in the '
                         'BN loss regularization used (--BN>0) are the global '
                         'mean and variance')

''' Configuration of basic GI '''
parser.add_argument('--n-seed', default=1, type=int,
                    help='Number of groups of pseudo-samples.')
parser.add_argument('--optim', default='Adam',
                    help='Optimizer for solving gradient inversion')
parser.add_argument('--gradient-loss-fun', default='sim', type=str,
                    choices=['sim', 'L2'],
                    help='Function of measuring lossbetween pseudo-gradients'
                         'and target-gradients. sim: negative cosine '
                         'similarity, L2: L2-norm')
parser.add_argument('--grad-sign', action='store_true',
                    help='Apply the sign of gradients of pseudo-samples to'
                         'update pseudo-samples, Great for negative cosine'
                         'similarity (gradient loss functions)')
parser.add_argument('--input-boxed', action='store_true',
                    help='Limit the input space forcefully')
parser.add_argument('--min-grads-loss', action='store_true',
                    help='The final result takes the sample with '
                         'the smallest gradient loss')

''' configuration of superparameters '''
parser.add_argument('--epochs', default=1, type=int,
                    help='Total epochs of running gradient inversion.')
parser.add_argument('--lr', default=0., type=float,
                    help='Initial update step size for running code.')
parser.add_argument('--grads-weight', default=1, type=float,
                    help='Weight of gradient loss (default: 1)')
parser.add_argument('--TV', default=0, type=float,
                    help='Weight of total variation regularization.')
parser.add_argument('--BN', default=0, type=float,
                    help='Weight of bn loss regularization.')
parser.add_argument('--input-norm', default=0, type=float,
                    help='Weight of norm of pseudo-samples (regularization)')

''' GI component selection '''
parser.add_argument('--pseudo-label-init', default='known', type=str,
                    choices=['from_grads', 'known'],
                    help='Way for initializing pseudo labels, from_grads: '
                         'Apply our proposed label reconstruction algorithm; '
                         'known: set the target sample to be known')
parser.add_argument('--superparameters-search', action='store_true',
                    help='Start the hyperparameter automatic search '
                         'algorithm, supported by ray-tune')
parser.add_argument('--MinCombine', action='store_true',
                    help='Use the Minimum Loss'
                         'Combinatorial Optimization')

''' Configuration of E2EGI '''
parser.add_argument('--Group', default=0.01, type=float,
                    help='Weight of group regularization.')
parser.add_argument('--T-init-rate', default=0.3, type=float,
                    help='Epochs of Initial Reconstruction')
parser.add_argument('--total-T-in-rate', default=0.3, type=float,
                    help='Total epochs of the Minimum Loss combinatorial '
                         'Optimization')
parser.add_argument('--T-in', default=1000, type=int,
                    help='Interval of constructing a new group consistency '
                         'regularization.')
parser.add_argument('--T-end-rate', default=0.4, type=float,
                    help='Epochs of Final Reconstruction')
parser.add_argument('--input-noise', default=0.2, type=float,
                    help='The degree to which random noise is introduced into'
                         'the pseudo-input during the update process')

''' Configuration of Superparameters Search'''
parser.add_argument('--simulation-checkpoint', type=str,
                    help='Path of simulation checkpoint for Superparameters '
                         'Search, checkpoint: you can use your own samples '
                         'to obtain by performing training.py')
parser.add_argument('--max-concurrent', default=16, type=int,
                    help='The number of parallel hyperparameter search '
                         'experiments')
parser.add_argument('--num-samples', default=64, type=int,
                    help='Total number of hyperparameter search experiments')
parser.add_argument('--ngpus-per-trial', default=0.5,
                    help='The number of GPU resources spent '
                         'on each hyperparameter search experiment')
parser.add_argument('--epochs-tune', action='store_true',
                    help='Search epochs (hyperparameter)')
parser.add_argument('--lr-tune', action='store_true',
                    help='Search update step size setup '
                         '(hyperparameter: --lr)')
parser.add_argument('--TV-tune', action='store_true',
                    help='Search weight of total variation regularization'
                         '(hyperparameter: --TV)')
parser.add_argument('--BN-tune', action='store_true',
                    help='Search weight of BN loss regularization '
                         '(hyperparameter: --BN)')
parser.add_argument('--input-norm-tune', action='store_true',
                    help='Search weight of input norm regularization'
                         '(hyperparameter: --input-norm)')
parser.add_argument('--grads-weight-tune', action='store_true',
                    help='Search weight of gradent loss '
                         '(hyperparameter: --grads-weight)')
parser.add_argument('--verbose', default=3, type=int,
                    help='Verbosity mode. 0 = silent, 1 = only status updates,'
                         '2 = status and brief trial results, '
                         '3 = status and detailed trial result')
parser.add_argument('--max-t', default=3000, type=int,
                    help='max epochs of hyperparameter search')
parser.add_argument('--min-t', default=2000, type=int,
                    help='At least the number of iterations each search '
                         'experiment must run (after min-t culling begins)')
parser.add_argument('--HP-epochs', default=5000, type=int,
                    help='Total epochs of running gradient inversion during '
                         'hyperparameter search')

''' metric (just for test, the target samples is known) '''
parser.add_argument('--metric', action='store_true',
                    help='Print the similarity between the target-samples and '
                         'the pseudo-samples (only used to test the effect of '
                         'gradient inversion, the target sample is required '
                         'to be known)')
parser.add_argument('--one-to-one-similarity', action='store_true',
                    help='Strictly follow the image order to measure the '
                         'similarity between the target-samples and '
                         'the pseudo-samples, if False (default), Each pseudo '
                         'sample has to measure the similarity with all '
                         'target samples, and take the optimal similarity as '
                         'the final value (this will cause more overhead)')
parser.add_argument('--GInfoR', action='store_true',
                    help='get the gradient information ratio of each sample, '
                         'this metric can indicate the gradient leakage risk '
                         'faced by each sample')

args = parser.parse_args()


def main(param_config=None):

    args.clock = datetime.now()

    if args.superparameters_search:
        set_superparameters(args, param_config)

    # distributed
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f'torch seed: {torch.initial_seed()}')

    # Create file to save all run output
    build_history(args)

    # load checkpoint
    model, bn_mean_list, bn_var_list, \
        dm, ds, target_gradient, metric_dict = load_checkpoint(args)

    model = set_distributed(model, args)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # initialize fake-samples
    x_pseudo_list = init_x_pseudo(args)
    y_pseudo = get_y_pseudo(args, target_gradient, metric_dict)
    if args.distributed:
        # Each GPU is on average responsible for
        # the reconstruction task of a fraction of the samples
        x_pseudo_list, y_pseudo = get_rank_samples(
            x_pseudo_list, y_pseudo, args)

    # set gpu
    x_pseudo_list = x_pseudo_list.cuda(args.gpu)
    y_pseudo = y_pseudo.cuda(args.gpu)
    target_gradient = list((grad.cuda(args.gpu) for grad in target_gradient))
    dm = dm.cuda(args.gpu)
    ds = ds.cuda(args.gpu)

    # track bn loss
    bn_loss_layers = None
    if args.BN > 0:
        bn_loss_layers = set_BN_regularization(
                            bn_mean_list,
                            bn_var_list,
                            model,
                            args)

    # run gradient inversion to reconstruct samples
    x_recon, cache = reconstruction(
        x_pseudo_list,
        y_pseudo,
        target_gradient,
        model,
        dm,
        ds,
        args,
        metric_dict=metric_dict,
        bn_loss_layers=bn_loss_layers)

    # save results and print metric
    if not args.superparameters_search:
        save_results_with_metric(x_recon, y_pseudo, dm, ds, metric_dict, args)


if __name__ == '__main__':

    if args.superparameters_search:
        config_bk = save_args(args)
        HP_config = ray_tune_main(main, args)
        update_args(args, HP_config, config_bk)

    main()
