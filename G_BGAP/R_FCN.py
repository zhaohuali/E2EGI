"""
模型bias可选
多层全连接网络
自动调参norm
修正参数约束
"""

import random
import os

import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def relu_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(
            m.weight, a=0, mode='fan_in', nonlinearity='relu')


class DenseModel(nn.Module):
    def __init__(
            self, n_neural_list, bias=False):
        super().__init__()

        self.n_neural_list_len = len(n_neural_list)

        assert self.n_neural_list_len > 1

        if self.n_neural_list_len > 2:
            self.dense = self._make_layer(
                n_neural_list[0: self.n_neural_list_len-1], bias)

        n_classes = n_neural_list[-1]
        self.output = nn.Linear(n_neural_list[-2], n_classes, bias=bias)

    def _make_layer(self, n_neural_list, bias):
        layers = []
        act = nn.ReLU()

        for i in range(1, len(n_neural_list)):
            dense_act = nn.Sequential(
                nn.Linear(n_neural_list[i-1], n_neural_list[i], bias=bias),
                act)
            layers.append(dense_act)

        return nn.Sequential(*layers)

    def forward(self, x):
        x_flatten = x.view(x.size(0), -1)

        out_list = []
        if self.n_neural_list_len == 2:
            out = self.output(x_flatten)
            out_list.append(out)

        out = x_flatten
        if self.n_neural_list_len > 2:
            for i in range(self.n_neural_list_len - 2):
                out = self.dense[i](out)
                out_list.append(out)
            out = self.output(out)
            out_list.append(out)

        return out_list


def get_target(BS, setup, path):

    tt = torchvision.transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(root=path, train=True, download=True)

    if BS == 1:
        train_sample = random.choice(list(trainset))
        x_target, y_target = train_sample
        x_target = tt(x_target).unsqueeze(0)
    else:
        train_sample = random.sample(list(trainset), BS)
        x_target, y_target = list(zip(*train_sample))
        x_target = [tt(im) for im in x_target]
        x_target = torch.stack(x_target)

    y_target = torch.as_tensor(y_target).view(BS, )
    print(f'y_target: {y_target}')

    assert x_target.shape == (BS, 1, 28, 28)
    assert y_target.shape == (BS, )

    return x_target.to(**setup), y_target


def get_gradients(x_target, y_target, model):

    CE_fn = nn.CrossEntropyLoss()
    target_out_list = model(x_target)
    last_output = target_out_list[-1]

    loss = CE_fn(last_output, y_target)

    target_gradent = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=False)
    grads = list([p.detach().clone() for p in target_gradent])

    target_out_list = list([t.detach().clone() for t in target_out_list])

    return grads, target_out_list


def auto_gradient_inversion_for_output(
        BS, n_classes, last_dW, last_W,
        setup, y_target_onehot, bias,
        norm_coeff_min_max=[-8, 0],
        grads_loss_fn='sim',
        known_out_list=None, ):

    _min, _max = norm_coeff_min_max
    norm_coeff_config = torch.logspace(_min, _max, steps=30).to(**setup)
    flags = torch.ones(size=norm_coeff_config.size())
    grads_list = torch.ones(size=norm_coeff_config.size()) * float('inf')

    mid_idx = len(norm_coeff_config) // 2
    min_idx = -1
    m = len(norm_coeff_config)
    pred_output_list = torch.zeros(size=(m, BS, n_classes))

    while(mid_idx >= 0 and mid_idx < m and min_idx != mid_idx):

        if min_idx != -1:
            mid_idx = min_idx
        print(f'mid_idx [{mid_idx}]')

        if flags[mid_idx] == 1:
            norm_coeff = norm_coeff_config[mid_idx]
            pred_output, grads_loss = gradient_inversion_for_output(
                BS, n_classes, last_dW, last_W,
                setup, y_target_onehot, bias,
                norm_coeff,
                grads_loss_fn=grads_loss_fn,
                known_out_list=known_out_list, )
            grads_list[mid_idx] = grads_loss
            flags[mid_idx] = 0
            pred_output_list[min_idx] = pred_output
            print(f'[{mid_idx}]: {grads_loss}')

        if mid_idx-1 > 0 and flags[mid_idx-1] == 1:
            norm_coeff = norm_coeff_config[mid_idx-1]
            pred_output, grads_loss = gradient_inversion_for_output(
                BS, n_classes, last_dW, last_W,
                setup, y_target_onehot, bias,
                norm_coeff,
                grads_loss_fn=grads_loss_fn,
                known_out_list=known_out_list, )
            grads_list[mid_idx-1] = grads_loss
            flags[mid_idx-1] = 0
            pred_output_list[mid_idx-1] = pred_output
            print(f'[{mid_idx-1}]: {grads_loss}')

        if mid_idx+1 < m and flags[mid_idx+1] == 1:
            norm_coeff = norm_coeff_config[mid_idx+1]
            pred_output, grads_loss = gradient_inversion_for_output(
                BS, n_classes, last_dW, last_W,
                setup, y_target_onehot, bias,
                norm_coeff,
                grads_loss_fn=grads_loss_fn,
                known_out_list=known_out_list, )
            grads_list[mid_idx+1] = grads_loss
            flags[mid_idx+1] = 0
            pred_output_list[mid_idx+1] = pred_output
            print(f'[{mid_idx+1}]: {grads_loss}')

        min_idx = mid_idx
        if mid_idx-1 > 0 and grads_list[mid_idx-1] < grads_list[mid_idx]:
            min_idx = mid_idx-1
        if mid_idx+1 < m and grads_list[mid_idx+1] < grads_list[min_idx]:
            min_idx = mid_idx+1

        print(f'min_idx [{min_idx}]: {grads_list[min_idx]}')

    return pred_output_list[min_idx], grads_list[min_idx]


def gradient_inversion_for_output(
        BS, n_classes, last_dW, last_W,
        setup, y_target_onehot, bias,
        norm_coeff,
        grads_loss_fn='L2',
        known_out_list=None, ):

    epochs = 10000
    lr = 0.01

    pred_output = torch.randn(size=(BS, n_classes)).to(**setup)

    if known_out_list is not None:
        known_output = known_out_list[-1]
        n_known_samples = len(known_output)
        pred_output[0: n_known_samples] = known_output

    pred_output.requires_grad_(True)

    optimizer = torch.optim.Adam([pred_output], lr=lr)
    softmax_fn = nn.Softmax(dim=1)
    grads_w_w = (last_dW * last_W).sum(dim=1)

    loss_fn = nn.MSELoss()

    for i in range(epochs):
        optimizer.zero_grad()
        p = softmax_fn(pred_output)
        pred_grads_w_w = ((p - y_target_onehot) * pred_output).mean(dim=0)

        loss = 0
        if grads_loss_fn == 'sim':
            pnorm = [0., 0.]
            grads_loss = 0.
            grads_loss -= (pred_grads_w_w * grads_w_w).sum()
            pnorm[0] += pred_grads_w_w.pow(2).sum()
            pnorm[1] += grads_w_w.pow(2).sum()
            grads_loss = 1 + grads_loss / pnorm[0].sqrt() / pnorm[1].sqrt()

        elif grads_loss_fn == 'L2':
            grads_loss = loss_fn(pred_grads_w_w, grads_w_w)

        loss += grads_loss

        if bias:
            db = last_dW[:, -1]
            pred_db = (p - y_target_onehot).mean(dim=0)
            db_loss = loss_fn(db, pred_db)
            loss += db_loss

        loss += norm_coeff * pred_output.norm()

        loss.backward()
        optimizer.step()

    pred_output = pred_output.detach().clone()
    if known_out_list is not None:
        n_known_samples = len(known_output)
        pred_output[0: n_known_samples] = known_output

    return pred_output, grads_loss.item()


def reconstruct_input(
        BS, n_pre_neural, n_post_neural,
        K, W, dW, output, setup,
        weights_constraint_abled=False,
        rank_print=True, bias=None, relu=False):

    assert K.shape == (BS, n_post_neural)
    assert output.shape == (BS, n_post_neural)
    assert W.shape == (n_post_neural, n_pre_neural)
    assert dW.shape == (n_post_neural, n_pre_neural)

    A = torch.zeros(size=(
        n_post_neural*n_pre_neural, BS*n_pre_neural)).to(**setup)
    for i in range(n_post_neural):
        for j in range(n_pre_neural):
            for k in range(BS):
                A[i*n_pre_neural+j, k*n_pre_neural+j] = K[k, i]

    if weights_constraint_abled:

        A_weights = torch.zeros(size=(BS*n_post_neural, BS*n_pre_neural))
        for i in range(n_post_neural):
            for j in range(n_pre_neural):
                for k in range(BS):
                    A_weights[
                        k*n_post_neural+i,
                        k*n_pre_neural:(k+1)*n_pre_neural] = W[i]

        A = torch.cat([A, A_weights], dim=0)
        assert A.shape == (
            n_post_neural*n_pre_neural + BS*n_post_neural, BS*n_pre_neural)

    b = dW.view(-1, 1) * BS
    if weights_constraint_abled:
        if bias is not None:
            output_minus_bias = output - bias.view(1, -1)
            if relu:
                output_minus_bias = output_minus_bias.clamp(
                    0, output_minus_bias.max())
            b_weights = output_minus_bias.view(-1, 1)
        else:
            b_weights = output.view(-1, 1)
        b = torch.cat([b, b_weights], dim=0)
        assert b.shape == (n_post_neural*n_pre_neural + BS*n_post_neural, 1)

        if relu:
            row, col = torch.where(b_weights == 0)
            A[n_post_neural*n_pre_neural + row, :] = 0

    A_numpy = A.numpy()
    b_numpy = b.numpy()
    pred_input, _, rank, _ = np.linalg.lstsq(A_numpy, b_numpy, rcond=None)

    pred_input = torch.as_tensor(pred_input).to(**setup)
    assert pred_input.shape == (BS * n_pre_neural, 1)
    pred_input = pred_input.view(BS, n_pre_neural)

    if rank_print:
        A_b = torch.cat([A, b], dim=1)
        rank_A = np.linalg.matrix_rank(A_numpy)
        A_b = A_b.numpy()
        rank_A_b = np.linalg.matrix_rank(A_b)
        cond_A = np.linalg.cond(A_numpy)
        print(f'A2.size: {A_numpy.shape}')
        print(f'rank[A]: {rank}, N[input]: {BS * n_pre_neural}')
        print(f'rank*[A]: {rank_A}, rank*[A|b]: {rank_A_b}, cond[A]: {cond_A}')

    return pred_input


def get_relu_gradients(pred_out, threshold=1e-3):

    # get gradients of ReLU
    ones = torch.ones(size=pred_out.size())
    zeros = torch.zeros(size=pred_out.size())
    drelu = torch.where(pred_out >= threshold, ones, zeros)
    assert drelu.shape == pred_out.shape

    return drelu


if __name__ == '__main__':

    torch.manual_seed(1)
    random.seed(1)
    BS = 4
    n_channels = 1
    data_w = 28
    data_h = 28
    n_features = data_w*data_h
    n_neural = 10
    n_classes = 10
    n_known_samples = 0
    norm_coeff = 5e-5
    threshold = 1e-3
    bias = True
    global_eval_mode = True
    layer_eval_mode = False
    weights_constraint_abled = False  # dont support for relu
    setup = {'device': 'cpu', 'dtype': torch.float32}
    loss_fn = nn.MSELoss()
    softmax_fn = nn.Softmax(dim=1)
    data_path = '/data/lzh/data/datasets/mnist'

    # build model
    n_neural_list = [n_features, n_neural, n_classes]
    model = DenseModel(n_neural_list, bias=bias).to(**setup)
    relu_weights_init(model)
    print(model)

    # build target
    x_target, y_target = get_target(BS, setup, path=data_path)
    assert x_target.shape == (BS, n_channels, data_w, data_h)
    assert y_target.shape == (BS, )

    # get gradients and output of model
    grads_list, target_out_list = get_gradients(x_target, y_target, model)

    # collect known conditions
    W_list = []
    b_list = [] if bias else None
    for key, value in model.state_dict().items():
        if key.endswith('weight'):
            W_list.append(value)
        elif key.endswith('bias'):
            b_list.append(value)

    dW_list = []
    db_list = [] if bias else None
    if bias:
        m = len(grads_list) // 2
        for i in range(m):
            dW_list.append(grads_list[i*2])
            db_list.append(grads_list[i*2+1])
    else:
        dW_list = grads_list

    y_target_onehot = F.one_hot(y_target, n_classes)
    assert y_target_onehot.shape == (BS, n_neural_list[-1])

    # for p and last output
    x_known = None
    known_out_list = None
    if n_known_samples > 0:
        x_known = x_target[0: n_known_samples]
        assert x_known.shape == (n_known_samples, n_channels, data_w, data_h)
        _known_out_list = model(x_known)
        known_out_list = list([p.detach().clone() for p in _known_out_list])

    if bias:
        last_W_b = torch.cat([W_list[-1], b_list[-1].view(-1, 1)], dim=1)
        last_dW_b = torch.cat([dW_list[-1], db_list[-1].view(-1, 1)], dim=1)
    else:
        last_W_b = W_list[-1]
        last_dW_b = dW_list[-1]

    if not global_eval_mode and not layer_eval_mode:
        pred_out, grads_loss = auto_gradient_inversion_for_output(
            BS, n_classes, last_dW_b, last_W_b,
            setup, y_target_onehot, bias,)
    #     pred_out = gradient_inversion_for_output(
    #         BS, n_classes, last_dW_b, last_W_b, setup,
    #         y_target_onehot, bias, norm_coeff,
    #         known_out_list=known_out_list)
    else:
        pred_out = target_out_list[-1]

    # metric of pred_out
    grads_w_w = (last_dW_b * last_W_b).sum(dim=1)
    target_out = target_out_list[-1]
    p = softmax_fn(pred_out)
    pred_grads_w_w = ((p - y_target_onehot) * pred_out).mean(dim=0)
    grads_loss = loss_fn(pred_grads_w_w, grads_w_w)
    print(f'grads loss: {grads_loss}')
    out_loss = loss_fn(pred_out, target_out)
    print(f'out loss: {out_loss}')
    p_loss = loss_fn(p, softmax_fn(target_out))
    print(f'p loss: {p_loss}')
    t_norm = target_out.norm()
    f_norm = pred_out.norm()
    print(f't norm: {t_norm}')
    print(f'f norm: {f_norm}')

    # reconstruct out2
    p_minus_y = (p - y_target_onehot)
    assert p_minus_y.shape == (BS, n_classes)
    K = p_minus_y

    indices = range(len(n_neural_list) - 1)
    for i_input in reversed(indices):
        W = W_list[i_input]
        dW = dW_list[i_input]
        b = None
        if bias:
            b = b_list[i_input]

        if i_input != indices[-1]:
            relu = True
        else:
            relu = False

        pred_out = reconstruct_input(
            BS, n_neural_list[i_input], n_neural_list[i_input+1],
            K, W, dW, pred_out, setup,
            bias=b, weights_constraint_abled=weights_constraint_abled,
            relu=relu
        )

        if i_input != 0:
            # metric of pred_out
            target_out = nn.ReLU()(target_out_list[i_input-1])
            out_loss = loss_fn(pred_out, target_out)
            print(f'[{i_input}] out loss: {out_loss}')

            if global_eval_mode or layer_eval_mode:
                threshold = target_out[
                    torch.where(target_out > 0)[0],
                    torch.where(target_out > 0)[1]].min()
                print(f'threshold: {threshold}')

            drelu = get_relu_gradients(pred_out, threshold)
            # metric of drelu
            ones = torch.ones(size=pred_out.size())
            drelu_target = torch.where(target_out == 0, target_out, ones)
            n_correct = torch.eq(drelu_target, drelu).sum()
            rate = n_correct*100. / float(drelu.numel())
            print(f'[{i_input}] drelu correct rate: {rate:2f}%')

            if layer_eval_mode:
                drelu = drelu_target
                pred_out = target_out

            K = torch.mm(K, W) * drelu
            pred_out = pred_out * drelu

        else:
            pred_x = pred_out.reshape(BS, -1)
            min_ = pred_x.min(dim=1)[0].view(BS, 1)
            max_ = pred_x.max(dim=1)[0].view(BS, 1)
            pred_x = (pred_x - min_)/(max_ - min_)
            pred_x = pred_x.view(BS, 1, 28, 28)
            # metric of pred_x
            x_loss = loss_fn(pred_x, x_target)
            print(f'x loss: {x_loss}')

    tp = torchvision.transforms.ToPILImage()
    for i in range(BS):
        plt.figure(f'orig {i}')
        plt.imshow(tp(x_target[i]))
        plt.axis('off')
        plt.savefig(os.path.join(os.getcwd(), f'orig {i}.png'))

        plt.figure(f'reconstructed {i}')
        plt.imshow(tp(pred_x[i].clamp(0, 1)))
        plt.axis('off')
        plt.savefig(os.path.join(os.getcwd(), f'R-reconstructed {i}.png'))
