
from collections import OrderedDict

import torchvision
import torch
from torch import nn


def get_layer_rank(layer, n_v, in_data):

    n_w = 0
    for param in layer.parameters():
        if param.requires_grad:
            n_w += param.numel()

    out = layer(in_data)
    n_x = len(in_data.view(-1,))
    n_z = len(out.view(-1, ))

    rank = n_x - n_w - n_z
    rank_with_v = n_x - n_w - n_z - n_v
    print(f'n_x:{n_x} n_z:{n_z} n_w:{n_w} n_v:{n_v} RA-i:{rank} RA-i with v:{rank_with_v}')

    n_v = max(max(n_z-n_x, 0) - max(rank, 0), 0) + n_v

    return n_v, out, rank_with_v

def get_identity_resblock_rank(block, n_v, in_data):

    layer = block.conv2

    out = block(in_data)
    n_x = len(in_data.view(-1,))
    n_z = len(out.view(-1, ))
    n_w = 0
    for param in layer.parameters():
        if param.requires_grad:
            n_w += param.numel()

    rank = n_x - n_w - n_z
    rank_with_v = n_x - n_w - n_z - n_v
    print(f'n_x:{n_x} n_z:{n_z} n_w:{n_w} n_v:{n_v} RA-i:{rank} RA-i with v:{rank_with_v}')

    n_v = max(max(n_z-n_x, 0) - max(rank, 0), 0) + n_v

    return n_v, out, rank_with_v

def get_conv_resblock_rank(block, n_v, in_data):

    out = block(in_data)
    n_x = len(in_data.view(-1,))
    n_z = len(out.view(-1, ))
    n_w = 0

    layer = block.conv2
    for param in layer.parameters():
        if param.requires_grad:
            n_w += param.numel()
    layer = block.downsample[0]
    for param in layer.parameters():
        if param.requires_grad:
            n_w += param.numel()

    rank = n_x - n_w - n_z
    rank_with_v = n_x - n_w - n_z - n_v
    print(f'n_x:{n_x} n_z:{n_z} n_w:{n_w} n_v:{n_v} RA-i:{rank} RA-i with v:{rank_with_v}')

    n_v = max(max(n_z-n_x, 0) - max(rank, 0), 0) + n_v

    return n_v, out, rank_with_v

def get_resnet_18_rank(model, input_size, device='cpu'):

    x = torch.randn(*input_size).to(device)

    rank_with_v_list = []
    n_v = 0

    print('[conv1]')
    conv1 = model.conv1
    n_v, x, rank_with_v = get_layer_rank(conv1, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[bn1]')
    bn1 = model.bn1
    x = bn1(x)

    print('[relu]')
    relu = model.relu
    x = relu(x)

    print('[maxpool]')
    maxpool = model.maxpool
    n_v, x, rank_with_v = get_layer_rank(maxpool, n_v, x)
    rank_with_v_list.append(rank_with_v)
    
    print('[layer1.0]')
    layer1_0 = model.layer1[0]
    n_v, x, rank_with_v = get_identity_resblock_rank(layer1_0, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[layer1_1]')
    layer1_1 = model.layer1[1]
    n_v, x, rank_with_v = get_identity_resblock_rank(layer1_1, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[layer2_0]')
    layer2_0 = model.layer2[0]
    n_v, x, rank_with_v = get_conv_resblock_rank(layer2_0, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[layer2_1]')
    layer2_1 = model.layer2[1]
    n_v, x, rank_with_v = get_identity_resblock_rank(layer2_1, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[layer3_0]')
    layer3_0 = model.layer3[0]
    n_v, x, rank_with_v = get_conv_resblock_rank(layer3_0, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[layer3_1]')
    layer3_1 = model.layer3[1]
    n_v, x, rank_with_v = get_identity_resblock_rank(layer3_1, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[layer4_0]')
    layer4_0 = model.layer4[0]
    n_v, x, rank_with_v = get_conv_resblock_rank(layer4_0, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[layer4_1]')
    layer4_1 = model.layer4[1]
    n_v, x, rank_with_v = get_identity_resblock_rank(layer4_1, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[avgpool]')
    avgpool = model.avgpool
    n_v, x, rank_with_v = get_layer_rank(avgpool, n_v, x)
    rank_with_v_list.append(rank_with_v)

    print('[fc]')
    fc = model.fc
    x = x.view(x.size(0), -1)
    n_v, x, rank_with_v = get_layer_rank(fc, n_v, x)
    rank_with_v_list.append(rank_with_v)

    max_rank_with_v = torch.as_tensor(rank_with_v_list).max()

    return max_rank_with_v

def get_ra_index(model, input_size, device='cpu'):

    x = torch.randn(*input_size).to(device)

    rank_with_v_list = []
    n_v = 0

    for name, layer in model.named_children():
        print(f'[layer] {name}')

        if (isinstance(layer, nn.Conv2d) or
                isinstance(layer, nn.MaxPool2d) or 
                isinstance(layer, nn.AdaptiveAvgPool2d)):
            n_v, x, rank_with_v = get_layer_rank(layer, n_v, x)
            rank_with_v_list.append(rank_with_v)
        elif (isinstance(layer, nn.BatchNorm2d) or 
                isinstance(layer, nn.ReLU)):
            x = layer(x)
        elif isinstance(layer, nn.Linear):
            x = x.view(x.size(0), -1)
            n_v, x, rank_with_v = get_layer_rank(layer, n_v, x)
            rank_with_v_list.append(rank_with_v)
        elif 'layer' in name:
            for name, sublayer in layer.named_children():
                if 'downsample' in dict(sublayer.named_children()).keys():
                    n_v, x, rank_with_v = get_conv_resblock_rank(sublayer, n_v, x)
                else:
                    n_v, x, rank_with_v = get_identity_resblock_rank(sublayer, n_v, x)
                rank_with_v_list.append(rank_with_v)
        
    max_rank_with_v = torch.as_tensor(rank_with_v_list).max()

    return max_rank_with_v


    