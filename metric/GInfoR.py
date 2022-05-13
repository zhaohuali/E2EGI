
import torch
import torch.nn as nn
import torch.nn.functional as F

from prettytable import PrettyTable


def get_all_pred_confi(model, x, model_eval=False):

    '''return: confi.shape=[bs, number of classes]'''
    if model_eval:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        out = model(x)
        confi = F.softmax(out, dim=-1).detach().clone().cpu()
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


def get_grad(model, x, y, idx=-1, model_eval=False):

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

    if idx >= 0:
        grads = torch.autograd.grad(model_loss[idx],
                                    model.parameters(),
                                    create_graph=False)
    else:
        grads = torch.autograd.grad(model_loss,
                                    model.parameters(),
                                    create_graph=False)

    grads = list([grad.detach().clone().cpu() for grad in grads])
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


def get_gir(model, metric_dict, model_eval):

    '''gradients info rate'''
    x = metric_dict['x_true']
    y = metric_dict['y_true']

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

    bs = x.size(0)
    true_pred, true_pred_confi = get_true_pred_confi(model, x, y, model_eval)
    max_pred, max_pred_confi = get_max_pred_confi(model, x, model_eval)

    total_grads = get_grad(model, x, y, idx=-1, model_eval=model_eval)
    total_elem = 0
    for tg in total_grads:
        total_elem += tg.nelement()

    total_grads_norm = torch.stack([g.norm() for g in total_grads]).mean()

    info_list = []
    gir_list = []
    for i in range(bs):

        gir = dict()

        gir['T:IDX'] = i
        gir['T:Label'] = true_pred[i]
        gir['T:confi'] = true_pred_confi[i]
        gir['Max:Label'] = max_pred[i]
        gir['Max:confi'] = max_pred_confi[i]

        single_grads = get_grad(model, x, y, idx=i, model_eval=model_eval)

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
        gir['G Sign Rate(%)'] = 0

        single_grads_norm = \
            torch.stack([g.norm() for g in single_grads]).mean()
        gir['G Norm'] = single_grads_norm

        gir_list.append(gir)

    info_list = torch.tensor(info_list)
    info_sum = info_list.sum()
    if info_list.min() < 0:
        info_sum += info_list.min().abs()*bs

    infoR_list = []
    for info in info_list:
        if info_list.min() < 0:
            infoR = (info + info_list.min().abs()) / info_sum
        else:
            infoR = info / info_sum
        infoR_list.append(infoR)
    for i in range(bs):
        gir_list[i]['GIR(%)'] = infoR_list[i] * 100

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
