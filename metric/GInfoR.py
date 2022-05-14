
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from prettytable import PrettyTable


def get_all_pred_confi(model, x, model_eval):

    '''return: confi.shape=[bs, number of classes]'''
    if model_eval:
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        out = model(x)
        print(f'out: {out.shape}')
        confi = F.softmax(out, dim=-1).detach().clone().cpu()
        print(f'confi: {confi.shape}')
    return confi


def get_max_pred_confi(model, x, model_eval):

    all_confi = get_all_pred_confi(model, x, model_eval)
    max_pred = all_confi.argmax(-1).view(-1,)
    pred_confi = [all_confi[i, pred].item() for i, pred in enumerate(max_pred)]
    return max_pred, pred_confi


def get_true_pred_confi(model, x, y, model_eval):

    all_confi = get_all_pred_confi(model, x, model_eval)
    y = y.view(-1,)
    pred_confi = [all_confi[i, t_y].item() for i, t_y in enumerate(y)]

    return y, pred_confi


def get_grad(model, x, y, args, idx=-1, model_eval=False):

    if model_eval:
        model.eval()
    else:
        model.train()

    model.zero_grad()
    fake_out = model(x)

    if idx >= 0:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')

    model_loss = criterion(fake_out, y)
    print(f'model_loss: {model_loss}')

    if idx >= 0:
        grads = torch.autograd.grad(model_loss[idx],
                                    model.parameters(),
                                    create_graph=False)
    else:
        grads = torch.autograd.grad(model_loss,
                                    model.parameters(),
                                    create_graph=False)

    grads = list([grad.detach().clone() for grad in grads])

    if args.distributed and idx == -1:
        average_gradients(grads, args.ngpus_per_node)

    return grads


def compute_sim_loss_and_sign_rate(grads1, grads2):

    sim_diff = 0
    n_sign = 0
    pnorm = [0, 0]
    for sg, tg in zip(grads1, grads2):
        sim_diff -= (sg * tg).sum()
        pnorm[0] += tg.pow(2).sum()
        pnorm[1] += sg.pow(2).sum()

        n_sign += (tg.sign() == sg.sign()).sum()

    sim_diff = 1 + sim_diff / pnorm[0].sqrt() / pnorm[1].sqrt()

    return sim_diff, n_sign


def compute_L2_loss(grads1, grads2):

    L2_diff = 0
    for sg, tg in zip(grads1, grads2):
        _diff = ((sg - tg) ** 2).sum()
        L2_diff += _diff

    return L2_diff


def get_rank_samples(x_pseudo, y_pseudo, bs):

    num_replicas = dist.get_world_size()
    rank = dist.get_rank()

    if bs % num_replicas != 0:
        raise ValueError('number of samples % world_size != 0')
    else:
        num_samples = math.ceil(bs / num_replicas)

    start_idx = int(rank * num_samples)
    end_idx = int((rank + 1) * num_samples)

    rank_x_pseudo = x_pseudo[start_idx: end_idx, :, :, :]
    rank_y_pseudo = y_pseudo[start_idx: end_idx]

    return rank_x_pseudo, rank_y_pseudo


def get_gir(model, metric_dict, args):

    '''gradients info rate'''
    x = metric_dict['x_true']
    y = metric_dict['y_true']
    bs = x.shape[0]
    gpu = args.gpu
    model_eval = args.model_eval

    if args.distributed:
        x, y = get_rank_samples(x, y, bs)
    x, y = x.to(gpu), y.to(gpu)
    bs = x.shape[0]
    print(x.shape)
    print(y.shape)

    column_name = [
        'T:IDX',
        'T:Label',
        'T:confi',
        'Max:Label',
        'Max:confi',
        'G Norm',
        'G L2',
        'G Sim',
        'G Sign Rate(%)',
        'G info',
        'GIR(%)']

    true_pred, true_pred_confi = get_true_pred_confi(
        model, x, y, model_eval)
    print(f'true_pred_confi: {true_pred_confi}')
    max_pred, max_pred_confi = get_max_pred_confi(model, x, model_eval)
    print(f'max_pred: {max_pred}')
    print(f'max_pred_confi: {max_pred_confi}')

    total_grads = get_grad(model, x, y, args, idx=-1, model_eval=model_eval)
    total_elem = 0
    for tg in total_grads:
        total_elem += tg.nelement()

    total_grads_norm = torch.stack([g.norm() for g in total_grads]).mean()
    print(f'total_grads_norm: {total_grads_norm}')

    info_list = []
    gir_list = []
    for i in range(bs):

        gir = dict()

        gir['T:IDX'] = i
        gir['T:Label'] = true_pred[i]
        gir['T:confi'] = true_pred_confi[i]
        gir['Max:Label'] = max_pred[i]
        gir['Max:confi'] = max_pred_confi[i]

        single_grads = get_grad(
            model, x, y, args, idx=i, model_eval=model_eval)

        info = 0
        for sg, tg in zip(single_grads, total_grads):
            info += (sg * tg).sum()
        gir['G info'] = info
        info_list.append(info)

        gir['G L2'] = compute_L2_loss(single_grads, total_grads)

        sim_diff, n_sign = compute_sim_loss_and_sign_rate(
            single_grads, total_grads)

        gir['G Sim'] = sim_diff
        gir['G Sign Rate(%)'] = int(n_sign * 100 / total_elem)

        single_grads_norm = \
            torch.stack([g.norm() for g in single_grads]).mean()
        print(f'single_grads_norm: {single_grads_norm}')
        gir['G Norm'] = single_grads_norm

        gir_list.append(gir)
        print(gir)

    info_list = torch.tensor(info_list)
    print(f'a: {info_list}')
    info_list_min = 0
    if args.distributed:
        info_list_gather = gather_distributed(info_list, gpu)
        info_sum = info_list_gather.sum()
        if info_list_gather.min() < 0:
            info_list_min = info_list_gather.min().abs()
            info_sum += info_list_min * bs
    else:
        info_sum = info_list.sum()
        if info_list.min() < 0:
            info_list_min = info_list.min().abs()
            info_sum += info_list_min * bs
    print(info_sum)

    infoR_list = []
    for info in info_list:
        infoR = (info + info_list_min) / info_sum
        infoR_list.append(infoR.item())
    for i in range(bs):
        gir_list[i]['GIR(%)'] = infoR_list[i] * 100
    print(f'infoR_list: {infoR_list}')
    print(f'infoR sum: {sum(infoR_list)}')

    table = PrettyTable(column_name)
    for i in range(bs):
        values = []
        for key in column_name:
            value = gir_list[i][key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, float):
                values.append(value)
            else:
                values.append(f'{value:.4f}')
        table.add_row(values)

    # for total grads
    total_grads_values = dict()
    total_grads_values['T:IDX'] = 'Target'
    total_grads_values['G Norm'] = total_grads_norm
    total_grads_values['G info'] = info_sum
    values = []
    for key in column_name:
        if key in total_grads_values:
            value = total_grads_values[key]
        else:
            value = '-'
        if isinstance(value, torch.Tensor):
            value = value.item()
        if not isinstance(value, float):
            values.append(value)
        else:
            values.append(f'{value:.4f}')
    table.add_row(values)

    return table


def average_gradients(gradent, ngpus_per_node):

    size = float(ngpus_per_node)
    for grad in gradent:
        dist.all_reduce(grad.data, op=dist.ReduceOp.SUM)
        grad.data /= size


def sum_distributed(data, gpu):

    data = torch.as_tensor(data).to(gpu)
    dist.all_reduce(data.data, op=dist.ReduceOp.SUM)
    return data


def gather_distributed(data, gpu):

    data = torch.as_tensor(data).to(gpu)
    n_ranks = dist.get_world_size()
    tensor_list = [
        torch.zeros(
            len(data),
            dtype=data.dtype).to(gpu)
        for _ in range(n_ranks)]
    dist.all_gather(tensor_list, data)

    tensor_list = torch.cat(tensor_list, dim=0)

    return tensor_list
