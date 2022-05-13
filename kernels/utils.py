
import torch


def simple_total_variation(x):
    """Anisotropic TV. 使得图像更平滑"""
    if x.dim() == 3:
        x.data = x.view(1, *x.size())
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
