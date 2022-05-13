
import torch
from torch.distributed import ReduceOp


def simple_total_variation(x):
    """Anisotropic TV. 使得图像更平滑"""
    if x.dim() == 3:
        x.data = x.view(1, *x.size())
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


class BNForwardLossHook():

    def __init__(
            self, module, distributed,
            bn_mean, bn_var, process_group=None):

        self.distributed = distributed
        self.process_group = process_group
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        if not self.distributed:
            mean, var = self.compute_normal_batchnorm_statistics(input[0])
        else:
            mean, var = self.compute_sync_batchnorm_statistics(input[0])

        self.bn_loss = torch.norm(self.bn_mean - mean, 2) + \
            torch.norm(self.bn_var - var, 2)

        # must have no output

    def compute_normal_batchnorm_statistics(self, x):

        channel_first_input = x.transpose(0, 1).contiguous()
        squashed_input_tensor_view = channel_first_input.view(
                channel_first_input.size(0), -1)
        local_mean = torch.mean(squashed_input_tensor_view, 1)
        local_sqr_mean = torch.pow(
                squashed_input_tensor_view, 2).mean(1)
        mean = local_mean
        var = local_sqr_mean - local_mean.pow(2)

        return mean, var

    def compute_sync_batchnorm_statistics(self, x):
        '''Code referenced from https://github.com/NVIDIA/apex/'''

        process_group = self.process_group
        world_size = 1
        if not self.process_group:
            process_group = torch.distributed.group.WORLD

        channel_first_input = x.transpose(0, 1).contiguous()
        squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)

        local_mean = torch.mean(squashed_input_tensor_view, 1)
        local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(process_group)
            torch.distributed.all_reduce(
                        local_mean, ReduceOp.SUM, process_group)
            mean = local_mean / world_size
            torch.distributed.all_reduce(
                        local_sqr_mean, ReduceOp.SUM, process_group)
            sqr_mean = local_sqr_mean / world_size
        else:
            raise ValueError('distributed not initialize')

        var = sqr_mean - mean.pow(2)

        return mean, var

    def close(self):
        self.hook.remove()


def compute_label_acc(y_true, y_fake):
    '''度量标签的匹配数目'''

    y_true_sort = y_true.cpu().view(-1,).sort()[0]
    y_fake_sort = y_fake.cpu().view(-1, ).sort()[0]

    i = 0
    j = 0
    n_true = len(y_true_sort)
    n_fake = len(y_fake_sort)
    n_correct = 0

    while i < n_true and j < n_fake:

        if y_true_sort[i] == y_fake_sort[j]:
            n_correct += 1
            i += 1
            j += 1
        elif y_true_sort[i] > y_fake_sort[j]:
            j += 1
        elif y_true_sort[i] < y_fake_sort[j]:
            i += 1
    return n_correct, n_correct / n_true
