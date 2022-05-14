
import os

import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

__all__ = ['kaiming_uniform', 'save_results', 'BNForwardFeatureHook']


class BNForwardFeatureHook():

    def __init__(self, module, distributed, process_group=None):
        self.distributed = distributed
        self.process_group = process_group
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        x = input[0]
        with torch.no_grad():
            channel_first_input = x.transpose(0, 1).contiguous()
            squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
            local_mean = torch.mean(squashed_input_tensor_view, 1)
            local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)
            self.mean = local_mean
            self.var = local_sqr_mean - local_mean.pow(2)

    def close(self):
        self.hook.remove()


def save_results(x, y, args):

    x = x.detach().clone().cpu()
    y = y.detach().clone().cpu()
    C = 3
    imgs = to_img(x)
    plt_imgs(imgs, y, C, title='F')
    path = os.path.join(
        args.results,
        f'id{args.id}_{args.data_name}_'
        + f'b{args.batch_size}_i{args.target_idx}'
        + '_true_imgs.jpg')
    plt.savefig(path)
    plt.close()
    print(f' > save image: {path}')


def to_img(x):
    '''
    x.shape = (n_imgs, n_channels, H, W)
    '''
    x = x.detach().clone().cpu()
    n_channels = x.shape[1]

    imgs = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    C = n_channels
    dm = torch.as_tensor(mean).view(C, 1, 1)
    ds = torch.as_tensor(std).view(C, 1, 1)
    for img in x:
        img = img * ds + dm
        img = img.clamp(0, 1)
        imgs.append(img)

    imgs = torch.stack(imgs)
    return imgs


def plt_imgs(imgs, y, n_channels, title=''):

    n_imgs = len(imgs)

    tt = transforms.ToPILImage()

    if n_imgs <= 8:
        plt.figure(figsize=(40, 24))
    else:
        plt.figure(figsize=(40, n_imgs))

    i = 0
    for img in imgs:
        plt.subplot(n_imgs*2 // 8 + 1, 8, i+1)
        if n_channels == 1:
            plt.imshow(tt(img), cmap='gray')
        else:
            plt.imshow(tt(img))
        plt.title(f'{title}:{y[i].item()}', fontsize=30)
        plt.axis('off')
        i += 1


def kaiming_uniform(m):
    torch.nn.init.kaiming_uniform_(
        m.weight, a=0, mode='fan_in', nonlinearity='relu')
