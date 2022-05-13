
import os
import numpy as np

import torch
import lpips
import matplotlib.pyplot as plt
import torch.distributed as dist
from prettytable import PrettyTable
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim


def save_imgs(
        x, y, dm, ds, distributed,
        input_size, rank, dir_path, clock, str_=''):

    if distributed:

        world_size = dist.get_world_size()
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        y_list = [torch.empty_like(y) for _ in range(world_size)]
        dist.all_gather(x_list, x)
        dist.all_gather(y_list, y)
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        assert x.shape == input_size, x.shape

    if not distributed or rank == 0:
        n_channels = input_size[1]
        # save pseudo-samples
        x = x.detach().clone().cpu()
        y = y.detach().clone().cpu()
        imgs = to_img(x, dm, ds)
        plt_imgs(imgs, y, n_channels, title='F')
        filename = f'{clock}_{str_}_pseudo.jpg'
        path = os.path.join(dir_path, filename)
        plt.savefig(path)
        plt.close()
        print(f' > save image: {path}')

    return x, y


def save_results_with_metric(x, y, dm, ds, metric_dict, args, str_=''):

    x, y = save_imgs(
        x, y, dm, ds, args.distributed, args.input_size,
        args.rank, args.path, args.clock, str_)

    if (not args.multiprocessing_distributed
            or (args.multiprocessing_distributed
                and args.rank % args.ngpus_per_node == 0)):

        n_channels = args.input_size[1]

        x_path = os.path.join(args.path, f'{args.clock}-x-pseudo.pth')
        y_path = os.path.join(args.path, f'{args.clock}-y-pseudo.pth')
        torch.save(x, x_path)
        torch.save(y, y_path)

        # save true-samples
        if args.metric:
            x_true = metric_dict['x_true'].cpu()
            y_true = metric_dict['y_true'].cpu()
            true_imgs = to_img(x_true, dm, ds)
            plt_imgs(true_imgs, y_true, n_channels, title='T')
            filename = f'{args.clock}-true .jpg'
            path = os.path.join(args.path, filename)
            plt.savefig(path)
            plt.close()
            print(f' > save image: {path}')

            compute_metric(x, x_true, dm, ds, args)


def compute_metric(x_recon, x_true, dm, ds, args):

    metric_list = ['mse', 'psnr', 'lpips', 'ssim']
    metric = ImageSimilarity(metric_list)

    print('inversion metric')
    imgs = to_img(x_recon, dm, ds)
    true_imgs = to_img(x_true, dm, ds)
    if args.one_to_one_similarity:
        if len(true_imgs) != len(imgs):
            print('[warning] number of true imgs != number of fake imgs')

        metrics_batch = metric.batch_similarity(
            true_imgs,
            imgs,
            T_to_R_map=torch.arange(0, min(len(true_imgs), len(imgs))))
    else:
        metrics_batch = metric.min_similarity(
            true_imgs,
            imgs)
    analysis_table = show_metric(metrics_batch)
    print(f'finish id: {args.id}\n'+str(analysis_table))


def show_metric(metrics_batch):
    column_name = ['T:IDX', 'mse', 'psnr', 'lpips', 'ssim']
    metric_name = column_name[1:]

    mean_metric = dict()
    for key in column_name:
        if key == 'T:IDX':
            mean_metric[key] = 'mean'
        else:
            mean_metric[key] = 0

    std_metric = dict()
    for key in column_name:
        if key == 'T:IDX':
            std_metric[key] = 'std'
        else:
            std_metric[key] = []

    for i, metrics in enumerate(metrics_batch):

        metrics_batch[i]['T:IDX'] = int(i)
        for key in metric_name:
            mean_metric[key] += metrics[key]
            std_metric[key].append(metrics[key])

    for key in metric_name:
        mean_metric[key] /= len(metrics_batch)
        std_metric[key] = torch.as_tensor(std_metric[key]).std().item()

    metrics_batch.append(mean_metric)
    metrics_batch.append(std_metric)

    table = PrettyTable(column_name)

    for i in range(len(metrics_batch)):
        values = []
        for key in column_name:
            value = metrics_batch[i][key]
            if not isinstance(value, float):
                values.append(value)
            else:
                values.append(f'{value:.4f}')
        table.add_row(values)

    return table


class ImageSimilarity():

    def __init__(self,
                 metric_list) -> None:

        self.metric_list = metric_list
        if 'lpips' in metric_list:
            self.lpips_fn = lpips.LPIPS(net='alex')

    def batch_similarity(
            self, target_batch,
            recon_batch, T_to_R_map):

        metrics_batch = []

        for j_R, i_T in enumerate(T_to_R_map):
            # 以重构样本顺序为正序
            target = target_batch[i_T]
            recon = recon_batch[j_R]
            metrics = self.compute_similarity(target, recon)
            metrics_batch.append(metrics)

        return metrics_batch

    def min_similarity(self, target_batch, recon_batch):

        metrics_batch = []

        for t_img in target_batch:

            if 'mse' in self.metric_list:
                min_mse = float('inf')
            if 'psnr' in self.metric_list:
                max_psnr = 0
            if 'lpips' in self.metric_list:
                min_lpips = float('inf')
            if 'ssim' in self.metric_list:
                max_ssim = 0

            for f_img in recon_batch:
                metrics = self.compute_similarity(t_img, f_img)

                if 'mse' in self.metric_list and metrics['mse'] < min_mse:
                    min_mse = metrics['mse']

                if 'psnr' in self.metric_list and metrics['psnr'] > max_psnr:
                    max_psnr = metrics['psnr']

                if 'lpips' in self.metric_list \
                        and metrics['lpips'] < min_lpips:
                    min_lpips = metrics['lpips']

                if 'ssim' in self.metric_list and metrics['ssim'] > max_ssim:
                    max_ssim = metrics['ssim']

            if 'mse' in self.metric_list:
                metrics['mse'] = min_mse
            if 'psnr' in self.metric_list:
                metrics['psnr'] = max_psnr
            if 'lpips' in self.metric_list:
                metrics['lpips'] = min_lpips
            if 'ssim' in self.metric_list:
                metrics['ssim'] = max_ssim

            metrics_batch.append(metrics)

        return metrics_batch

    def match_batch_based_label(self, target_y_batch, recon_y_batch):
        '''适用于唯一标签场景，获得目标样本与重构样本的标签匹配项'''
        T_to_R_map = []
        for y in target_y_batch:
            T_to_R_map.append(torch.where(recon_y_batch == y)[0].item())
        return T_to_R_map

    def compute_similarity(self, target, recon):

        metrics = {}

        if 'mse' in self.metric_list:
            metrics['mse'] = self.mean_squared_error(target, recon)

        if 'psnr' in self.metric_list:
            if 'mse' in self.metric_list:
                metrics['psnr'] = \
                    self.peak_signal_noise_ratio(metrics['mse'])
            else:
                mse = self.mean_squared_error(target, recon)
                metrics['psnr'] = \
                    self.peak_signal_noise_ratio(mse)

        if 'lpips' in self.metric_list:
            metrics['lpips'] = self.perceptual_metric(target, recon)

        if 'ssim' in self.metric_list:
            metrics['ssim'] = self.calc_ssim(target, recon)

        return metrics

    def mean_squared_error(self, target, recon):

        mse = ((target - recon) ** 2).mean()

        return mse.item()

    def peak_signal_noise_ratio(self, mse, factor=1.0):

        mse = torch.as_tensor(mse)
        return (10 * torch.log10(factor**2 / mse)).item()

    def perceptual_metric(self, target, recon):
        """LPIPS: https://github.com/richzhang/PerceptualSimilarity"""
        if target.size()[-1] < 32:
            # 该测量要求输入像素点大于32
            resize_fun = transforms.Resize(32)
        else:
            resize_fun = transforms.Lambda(lambda x: x)

        resizer = transforms.Compose([
            transforms.ToPILImage(),
            resize_fun,
            transforms.ToTensor(),
            transforms.Normalize(
                std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))])
        target = resizer(target)
        recon = resizer(recon)

        lpips_score = self.lpips_fn(target, recon)

        return lpips_score.item()

    def calc_ssim(self, target, recon):
        '''tructural similarity index'''

        if target.shape[0] > 1:
            multichannel = True
        else:
            multichannel = False
        tt = transforms.ToPILImage()
        target_np = np.array(tt(target))
        recon_np = np.array(tt(recon))
        ssim_score = ssim(
            target_np, recon_np,
            data_range=255, multichannel=multichannel)

        return ssim_score


def to_img(x, dm, ds):
    '''
    x.shape = (n_imgs, n_channels, H, W)
    '''
    x = x.detach().clone().cpu()
    ds = ds.cpu()
    dm = dm.cpu()

    imgs = []
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
