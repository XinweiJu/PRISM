# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img, input_mask=None):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)    
    
    if input_mask is not None:
        input_mask = input_mask.unsqueeze(dim=1)
        grad_disp_x = grad_disp_x[input_mask[:, :, :, :-1]]
        grad_disp_y = grad_disp_y[input_mask[:, :, :-1, :]]

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def normalize_grid(grid, H, W, align_corners):
    """ 归一化 grid 以匹配 PyTorch 的 grid_sample """
    if align_corners:
        x = ((grid[..., 0] + 1) / 2) * (W - 1)
        y = ((grid[..., 1] + 1) / 2) * (H - 1)
    else:
        x = ((grid[..., 0] + 1) * W - 1) / 2
        y = ((grid[..., 1] + 1) * H - 1) / 2
    return x, y

def apply_padding(image, x, y, padding_mode):
    N, C, H, W = image.shape

    if padding_mode == "border":
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)
    elif padding_mode == "reflection":
        x = torch.abs(x)
        y = torch.abs(y)
        x = torch.where(x >= W, 2 * (W - 1) - x, x)
        y = torch.where(y >= H, 2 * (H - 1) - y, y)
    else:  # padding_mode == "zeros"
        mask = (x < 0) | (x >= W) | (y < 0) | (y >= H)
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)
        return x, y, mask

    return x, y, None

import torch

def custom_grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True):
    """
    手写 PyTorch 版 grid_sample，支持 align_corners 和 padding_mode
    :param image: (N, C, H, W) 输入图像
    :param grid:  (N, H_out, W_out, 2) 归一化采样网格
    :param mode:  "bilinear" or "nearest"
    :param padding_mode: "zeros", "border", "reflection"
    :param align_corners: 是否使用 align_corners 归一化
    :return: (N, C, H_out, W_out) 插值后的结果
    """

    # image = image.to(torch.float64)  # 确保计算稳定
    # grid = grid.to(torch.float64)

    N, C, H, W = image.shape
    _, H_out, W_out, _ = grid.shape

    # 1. 归一化 grid (-1,1) -> 真实像素坐标
    x, y = normalize_grid(grid, H, W, align_corners)
    # print("normalized grid, x.shape, y.shape:", x.shape, y.shape)  #x.shape, y.shape: torch.Size([12, 288, 288]) torch.Size([12, 288, 288])

    # 2. 处理边界填充
    x, y, mask = apply_padding(image, x, y, padding_mode)
    # print("padded grid, x.shape, y.shape:", x.shape, y.shape)  #x.shape, y.shape: torch.Size([12, 288, 288]) torch.Size([12, 288, 288])

    # 3. 计算最近的四个像素点
    x0 = torch.floor(x).long()
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = torch.floor(y).long()
    y1 = (y0 + 1).clamp(0, H - 1)

        # 4. 计算插值权重
    tx = x - x0.float()
    ty = y - y0.float()

    tx = tx.unsqueeze(1)  # 变成 (12, 1, 288, 288)
    ty = ty.unsqueeze(1)  # 变成 (12, 1, 288, 288)


    # 5. 正确索引像素值，避免错误广播               
    y0 = y0.unsqueeze(1)  # 变成 (N, 1, H_out, W_out)
    x0 = x0.unsqueeze(1)

    y1 = y1.unsqueeze(1)
    x1 = x1.unsqueeze(1)

    I00 = image.gather(2, y0.expand(-1, C, -1, -1)).gather(3, x0.expand(-1, C, -1, -1))
    I10 = image.gather(2, y0.expand(-1, C, -1, -1)).gather(3, x1.expand(-1, C, -1, -1))
    I01 = image.gather(2, y1.expand(-1, C, -1, -1)).gather(3, x0.expand(-1, C, -1, -1))
    I11 = image.gather(2, y1.expand(-1, C, -1, -1)).gather(3, x1.expand(-1, C, -1, -1))

    # print("I00 shape:", I00.shape)
    # print("I10 shape:", I10.shape)
    # print("tx shape:", tx.shape)
    # print("ty shape:", ty.shape)

    # I00 shape: torch.Size([12, 3, 288, 288])
    # I10 shape: torch.Size([12, 3, 288, 288])
    # tx shape: torch.Size([12, 288, 288])
    # ty shape: torch.Size([12, 288, 288])

    # 6. 双线性插值
    a = (1 - tx) * I00 + tx * I10
    b = (1 - tx) * I01 + tx * I11
    sampled = (1 - ty) * a + ty * b
    # print("sampled shape:", sampled.shape)

    return sampled#.to(torch.float32)
