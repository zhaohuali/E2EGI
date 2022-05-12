
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

import argparse

import torchvision.models as models

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
parser.add_argument('data', metavar='DIR',
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

parser.add_argument('--results', default='', type=str,
                    help='path to store results')

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

args = parser.parse_args(args=[])
