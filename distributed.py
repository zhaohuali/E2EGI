
import math

import apex
import torch
import torch.distributed as dist


def get_rank_samples(x_pseudo_list, y_pseudo, args):

    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    bs = args.batch_size

    if bs % num_replicas != 0:
        raise ValueError('number of samples % world_size != 0')
    else:
        num_samples = math.ceil(bs / num_replicas)

    start_idx = int(rank * num_samples)
    end_idx = int((rank + 1) * num_samples)

    rank_x_pseudo_list = x_pseudo_list[:, start_idx: end_idx, :, :, :]
    rank_y_pseudo = y_pseudo[start_idx: end_idx]

    return rank_x_pseudo_list, rank_y_pseudo


def set_distributed(model, args):

    model = apex.parallel.convert_syncbn_model(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size
            # to all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size
        # to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model
