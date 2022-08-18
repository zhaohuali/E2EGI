
r"""
File for obtaining model gradient and other relevant information

The purpose of this file is to simulate the clients of federated learning
to train the model locally, and save it in the checkpoint to contain
public information and related information such as model gradients
that need to be uploaded to the parameter server.

Checkpoint includes (Except for public information):
- batch_size
- target_gradient: Calculated from the client's local data (the input sample)
- bn_mean_list: The mean of the BatchNorm layer obtained
    from the local data of the participants(optional,
    for training tasks based on synchronous batchnorm)
- bn_var_list: The variance of the BatchNorm layer ...
- x_true: Features of the input sample (just for testing,
    to verify the effect of E2EGI)
- y_true: Labels of the input sample (just for testing)
"""
__version__ = '0.1.0'

import os
import sys
import math
import random
import builtins
import argparse
from datetime import datetime

import apex
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
import torch.backends.cudnn as cudnn
from config import Logger

from train import get_target_samples, dataset_config, get_mean_std
from train import kaiming_uniform, save_results, BNForwardFeatureHook
from RGAP import get_ra_index


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')

''' Configuration of the target environment '''
parser.add_argument('--id', default='test', type=str, help='code ID')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

# the input sample
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data-name', default='imagenet', type=str)
parser.add_argument('--data-backup', default=None, type=str,
                    help='path to target samples from imagenet \
                        for same results')
parser.add_argument('--target-idx', default=0, type=int,
                    help='The index of the target sample')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# model
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained model')
parser.add_argument('--outlayer-state', default='normal', type=str)
parser.add_argument('--model-eval', action='store_true',
                    help='True: model.eval(), False:model.train()')
parser.add_argument('--upload-bn', action='store_true',
                    help='upload BN statistics (input mean and var)')

parser.add_argument('--results', default='./train/checkpoint', type=str,
                    help='path to store results')

# differential privacy
parser.add_argument('--enable-dp', action="store_true",
                    help="ensable privacy training and dont just train with vanilla SGD")
parser.add_argument('--sigma', type=float, default=None, help="Noise multiplier")
parser.add_argument('-C', '--max-per-sample-grad-norm', type=float, default=None, 
                    help="Clip per-sample gradients to this norm")
parser.add_argument('--delta', type=float, default=1e-5, help="Target delta")

# Rank Analysis
parser.add_argument('--ra', action='store_true', 
                    help='enable rank analysis of R-GAP')

''' Configuration of Hardware (GPUs)'''
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node '
                         'or multi node data parallel training')

args = parser.parse_args()


def main():

    # The parameters of all layers of the model are trained
    args.free_last_layers_list = ['all']

    # output environment configuration
    sys.stdout = Logger(args.results)
    args.clock = f'{datetime.now()}'
    print(f'\n========== {args.clock} ========')
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

    # set random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        print(f'torch seed: {torch.initial_seed()}')

    # get input samples
    x_true, y_true = get_target_samples(args)

    # distributed
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, x_true, y_true, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, x_true, y_true, args)


def main_worker(gpu, ngpus_per_node, x_true, y_true, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        print(f'torch seed: {torch.initial_seed()}')

    # create model
    model = models.__dict__[args.arch](pretrained=False)
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            if 'moco' not in args.pretrained:
                if args.pretrained.endswith('tar'):
                    checkpoint = torch.load(
                        args.pretrained, map_location='cpu')
                    state_dict = checkpoint['state_dict']
                    for k in list(state_dict.keys()):
                        if k.startswith('module.'):
                            state_dict[k[len("module."):]] = state_dict[k]
                        del state_dict[k]
                    model.load_state_dict(state_dict)
                elif args.pretrained.endswith('pth'):
                    model.load_state_dict(torch.load(args.pretrained))
                else:
                    raise ValueError('args.pretrained file naming format \
                                     is incorrect, should be \
                                     .pth (parameters) or .tar (checkpoint)')
            else:
                print("loading MoCoV2 checkpoint'{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if (k.startswith('module.encoder_q')
                            and not k.startswith('module.encoder_q.fc')):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = \
                            state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no model found at '{}'".format(args.pretrained))

    n_channels, W, H, n_classes = dataset_config(args.data_name)
    if n_classes != 1000:
        if 'resnet' in args.arch or 'regnet' in args.arch:
            model.fc = torch.nn.Linear(
                in_features=model.fc.in_features,
                out_features=n_classes)
        elif 'vgg' in args.arch:
            model.classifier[6] = nn.Linear(
                in_features=model.classifier[6].in_features,
                out_features=n_classes)

    if args.outlayer_state == 'normal':
        if 'resnet' in args.arch or 'regnet' in args.arch:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        elif 'vgg' in args.arch:
            model.classifier[6].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[6].bias.data.zero_()
    elif args.outlayer_state == 'kaiming_uniform':
        if 'resnet' in args.arch or 'regnet' in args.arch:
            kaiming_uniform(model.fc)
        elif 'vgg' in args.arch:
            kaiming_uniform(model.classifier[6])

    if args.ra:
        if 'resnet' not in args.arch:
            raise ValueError(f'no support model:{args.arch}, just support resnet')
        # todo: Rank Rnalysis
        input_size = (args.batch_size, n_channels, W, H)
        ra_i = get_ra_index(model, input_size)
        print(f'[model]{args.arch} [RA-i] {ra_i}')
        return 

    if (not args.multiprocessing_distributed
            or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):
        # save target samples
        print(f'x_true.shape: {x_true.shape}')
        print(f'y_true.shape: {y_true.shape}')
        save_results(x_true, y_true, args)

        # save target model
        model_path = os.path.join(args.results, 'model.pth')
        torch.save(model.state_dict(), model_path)
        print(f'save model: {model_path}')

    # support for SyncBatchNorm
    model = apex.parallel.convert_syncbn_model(model)

    # distributed
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.workers = int((args.workers+ngpus_per_node-1)/ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and
            # allocate batch_size to all available GPUs
            # if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and
        # allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    if args.upload_bn:
        bn_mean_var_layers = []
        for module in model.modules():
            if isinstance(module, apex.parallel.SyncBatchNorm):
                bn_mean_var_layers.append(BNForwardFeatureHook(
                    module, args.distributed))

    # load data
    if args.distributed:
        images, target = get_rank_samples(x_true, y_true)
    else:
        images, target = x_true, y_true
    images = images.cuda(args.gpu)
    target = target.cuda(args.gpu)

    # Set model layer parameters for training
    if 'all' not in args.free_last_layers_list:
        for name, param in model.named_parameters():
            for freed_name in args.free_last_layers_list:
                if freed_name in name:
                    param.requires_grad = True
                    print(f'[freed layers] {name}')
                    break
                else:
                    param.requires_grad = False

    # switch to mode
    if args.model_eval:
        model.eval()
    else:
        model.train()

    # compute output
    output = model(images)
    loss = criterion(output, target)

    # compute gradient and save results
    model.zero_grad()
    loss.backward()

    mean_list = None
    var_list = None
    if args.upload_bn:
        mean_list = [mod.mean.detach().clone() for mod in bn_mean_var_layers]
        var_list = [mod.var.detach().clone() for mod in bn_mean_var_layers]

    if (not args.multiprocessing_distributed
            or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):

        if args.enable_dp and args.max_per_sample_grad_norm is not None:
            # todo: gradient clipping
            total_norm = torch.nn.utils.clip_grad.clip_grad_norm(
                model.parameters(), max_norm=args.max_per_sample_grad_norm)
            print(f'run differential privacy, max_norm={args.max_per_sample_grad_norm}')
            print(f'orig norm is {total_norm:e}.')

        true_grads = list(
            (param.grad.detach().clone()
                for param in filter(
                lambda p: p.requires_grad, model.parameters())))

        # todo: add noise to gradients
        if args.enable_dp and args.sigma is not None:
            for i, g in enumerate(true_grads):
                noise = torch.normal(
                            mean=0,
                            std=args.sigma,
                            size=g.size(),
                            device=g.device)
                true_grads[i] += noise
            print(f'run differential privacy, sigma={args.sigma}')
            total_norm = torch.norm(torch.stack([torch.norm(
                (g.detach()), 2).cuda(args.gpu) for g in true_grads]), 2)
            print(f'new norm is {total_norm:e}.')

        
        # full_norm = torch.stack([g.norm() for g in true_grads]).mean()
        full_norm = torch.norm(torch.stack([torch.norm(
            (g.detach()), 2).cuda(args.gpu) for g in true_grads]), 2)
        print(f'Num of grads: {len(true_grads)}')
        print(f'Full gradient norm is {full_norm:e}.')

        mean, std = get_mean_std(args.data_name)
        dataset_setup = dict(mean=mean,
                             std=std,
                             size=(n_channels, W, H),
                             n_classes=n_classes)

        save_checkpoint({
                'clock': args.clock,
                'batch_size': args.batch_size,
                'dataset_setup': dataset_setup,
                'arch': args.arch,
                'model_eval': args.model_eval,
                'state_dict': torch.load(model_path),
                'free_last_layers_list': args.free_last_layers_list,
                'target_gradient': true_grads,
                'bn_mean_list': mean_list,
                'bn_var_list': var_list,
                'x_true': x_true,
                'y_true': y_true,
                'args': args
            }, args)


def save_checkpoint(state, args):

    model_file = os.path.basename(args.pretrained)
    filename = f'id{args.id}_{args.arch}_{model_file}'\
        + f'_{args.data_name}_b{args.batch_size}_i{args.target_idx}-'\
        + 'checkpoint.pth.tar'
    filepath = os.path.join(args.results, filename)
    torch.save(state, filepath)

    print(f'save checkpoint: {os.path.join(os.getcwd(), filename)}')


def get_rank_samples(x_true, y_true):

    num_replicas = dist.get_world_size()
    rank = dist.get_rank()

    if len(x_true) % num_replicas != 0:
        raise ValueError('number of samples % world_size != 0')
    else:
        num_samples = math.ceil(len(x_true) / num_replicas)

    start_idx = int(rank * num_samples)
    end_idx = int((rank + 1) * num_samples)

    return x_true[start_idx: end_idx, :, :, :], y_true[start_idx:end_idx]


class Logger(object):
    def __init__(self, results, filename='history.log', stream=sys.stdout):
        self.terminal = stream
        filepath = os.path.join(results, filename)
        self.log = open(filepath, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    main()
