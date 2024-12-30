from copy import deepcopy
from typing import Optional, Union, Dict, Any
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np
import math

"""
我们实现两个功能
    1. 计算single turbine的effective U，both in 2d and 3d
    2. 按照nD的距离截取整个风电场的speed profiles

两种思路：
    1. 考虑在扫掠面上插值到较为密集的网格，并逐点计算是否在风机扫掠面内
    2. 考虑粗糙网格的每个点，假如整个网格都在扫掠面内，则全部计入速度，
    如果部分不在，那就乘以一定的权重（等于扫掠面覆盖的面积比例），
    假如全都不在则不计入速度平均值计算
"""
Search_X_Relative = 200
Search_Y_Relative = 160
Search_Z_From = 20
Search_Z_To = 160
Cell_Size_origin = 20
Sweap_Radius = 63
Hub_height = 90
Upsample_scale = 16
Cell_Size_after = Cell_Size_origin / Upsample_scale


# Y_relative = Search_Y_Relative // Cell_Size
# Z_FROM = (Search_Z_From - Cell_Size) // Cell_Size
# Z_TO = (Search_Z_To - Cell_Size) // Cell_Size


def upsample_data_3d(data: torch.Tensor, scale_factor=16) -> torch.Tensor:
    """
    上采样数据到更高分辨率。

    参数:
    - data (torch.Tensor): 原始数据，形状为 [C, Z, H, W]
    - scale_factor (int): 上采样比例，默认为16（20m -> 1.25m）

    返回:
    - upsampled_data (torch.Tensor): 上采样后的数据
    """
    # 添加批次维度，形状变为 [1, C, Z, H, W], 以适应 interpolate 函数
    data = data.unsqueeze(0)

    # 使用三维插值，插值模式为线性
    upsampled_data = F.interpolate(
        data, scale_factor=(scale_factor, scale_factor, scale_factor), mode="trilinear", align_corners=False
    )

    # 移除批次维度，恢复为 [C, Z_new, H_new]
    upsampled_data = upsampled_data.squeeze(0)

    return upsampled_data


def is_point_in_swept_area_direct_batch(
    x_flat: torch.Tensor,
    y_flat: torch.Tensor,
    z_flat: torch.Tensor,
    turbine_info: torch.Tensor,
    radius,
    tolerance,
    Roter_Tower_D_Correction=70,
):
    """
    直接计算法批量处理：判断多个点是否在风机的扫掠面内

    参数:
    - x_flat: [N, ]: x 坐标
    - y_flat: [N, ]: y 坐标
    - z_flat: [N, ]: z 坐标
    - turbine_info (torch.Tensor): 风机信息 [x_location, y_location, z_location, yaw]
    - radius (float): 风机扫掠半径
    - tolerance (float): 距离容差
    - Roter_Tower_D_Correction (float): 风机rotor到tower的距离（rotor和实际location的距离）

    返回:
    - mask (torch.Tensor): 布尔型张量，长度为 N，指示每个点是否在扫掠面内
    """
    # print(f"inside func <is_point_in_swept_area_direct_batch>, turbine_info: {turbine_info}")
    x_loc, y_loc, yaw, _ = turbine_info.tolist()
    z_loc = Hub_height

    # 计算风轮法向量
    yaw_rad = math.radians(yaw)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # print(f"inside func <is_point_in_swept_area_direct_batch>, cos_yaw: {cos_yaw}, sin_yaw: {sin_yaw}")

    """
    # 计算点到风轮平面的距离 D = |(C - P) ⋅ N| = (PC·N)
    # 我们只要正的距离, 负的对应在下风口，我们不考虑
    # 由于不止一个点，我们实际得到的是一个向量(相当于每个向量都与同一个风轮法向量做点乘)
    x_flat:[N,]    x_loc:scalar    cos_yaw:scalar
    """
    # D: [N, ]: 点到风轮平面的距离
    D = (x_loc - x_flat) * cos_yaw + (y_loc - y_flat) * sin_yaw

    # 第一步判断：是否在风轮平面附近
    # mask_plane: [N, ]: 是否在风轮平面附近
    mask_plane: torch.Tensor = (D > Roter_Tower_D_Correction) & (D <= tolerance + Roter_Tower_D_Correction)

    # 对于在风轮平面内的点，计算投影点坐标
    # P_proj = P + DN
    # 注意x,y,z_proj的长度在mask索引之后不一样，应该小于N
    x_proj = x_flat[mask_plane] + D[mask_plane] * cos_yaw
    y_proj = y_flat[mask_plane] + D[mask_plane] * sin_yaw
    z_proj = z_flat[mask_plane]  # z 坐标不变

    # 计算投影点到风轮中心的距离 r
    r = torch.sqrt((x_proj - x_loc) ** 2 + (y_proj - y_loc) ** 2 + (z_proj - z_loc) ** 2)

    # 第二步判断：是否在扫掠圆盘内
    mask_radius = r <= radius

    # 初始化掩码，全为 False
    mask = torch.zeros_like(x_flat, dtype=torch.bool)
    # 将满足条件的点标记为 True
    mask[mask_plane] = mask_radius

    return mask


def calculate_effective_3d(
    turbine_info: torch.Tensor,
    data_tensor: torch.Tensor,
    debugging_mode: bool = False,
):
    """Calculate the effective U for a single turbine
    1. clip the data_tensor around the turbine according to the turbine_info
    2. Upsample the clip_data_tensor to a higher resolution
    3. Traverse all the data point in the clip_data_tensor:
        3.1. if current point is not inside the turbine swept surface, drop it
        3.2. if current point is near the front of turbine swept surface, include it in the mean velocity calculation
    4. Calculate the velocity component perpendicular to the turbine rotor plane

    Args:
        turbine_info (torch.Tensor): [4, ]: [x_location, y_location, yaw, is_wake_affected]
        data_tensor (torch.Tensor): [C, Z, H, W]
    """

    C, Z, H, W = data_tensor.shape
    x_loc, y_loc, yaw, is_wake_affected = turbine_info.tolist()

    # 根据风机位置裁剪数据，我们只需要考虑风机扫掠面大一点的区域
    # 坐标
    X_from_coord = x_loc - Search_X_Relative
    X_to_coord = x_loc + Search_X_Relative
    Y_from_coord = y_loc - Search_Y_Relative
    Y_to_coord = y_loc + Search_Y_Relative
    Z_from_coord = Search_Z_From
    Z_to_coord = Search_Z_To
    # 换算到20m网格的索引
    X_from_idx_origin = int(X_from_coord // Cell_Size_origin)
    X_to_idx_origin = int(X_to_coord // Cell_Size_origin + 1)
    Y_from_idx_origin = int(Y_from_coord // Cell_Size_origin)
    Y_to_idx_origin = int(Y_to_coord // Cell_Size_origin + 1)
    Z_from_idx_origin = int((Z_from_coord - Cell_Size_origin) // Cell_Size_origin)
    Z_to_idx_origin = int((Z_to_coord - Cell_Size_origin) // Cell_Size_origin + 1)

    # print(f"type of Y_from_idx_origin: {type(Y_from_idx_origin)}")

    # 截取裁剪后的数据，形状为 [C, Z_clip, H_clip, W_clip]
    clip_data_tensor = data_tensor[
        :, Z_from_idx_origin:Z_to_idx_origin, Y_from_idx_origin:Y_to_idx_origin, X_from_idx_origin:X_to_idx_origin
    ]

    # 对裁剪后的数据进行上采样
    scale_factor = Upsample_scale
    upsampled_clip_data_tensor = upsample_data_3d(clip_data_tensor, scale_factor=scale_factor)
    C, Z_up, H_up, W_up = upsampled_clip_data_tensor.shape

    # 获取上采样后每个数据点的坐标
    x_coords = torch.linspace(X_from_coord, X_to_coord, W_up)
    y_coords = torch.linspace(Y_from_coord, Y_to_coord, H_up)
    z_coords = torch.linspace(Z_from_coord, Z_to_coord, Z_up)

    # 创建网格
    z_grid, y_grid, x_grid = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    # 展平
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()

    # 计算是否在扫掠面内
    mask = is_point_in_swept_area_direct_batch(
        x_flat, y_flat, z_flat, turbine_info, Sweap_Radius, tolerance=Cell_Size_after * 64
    )

    if not mask.any():
        # 如果没有点在扫掠面内，返回零向量或适当的默认值
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    # 获取在扫掠面内的风速
    swept_U = upsampled_clip_data_tensor[0, :, :, :].flatten()[mask]
    swept_V = upsampled_clip_data_tensor[1, :, :, :].flatten()[mask]
    # swept_W = upsampled_clip_data_tensor[2, :, :, :].flatten()[mask]

    # 计算每个点的有效风速分量(即平行风轮法向量的风速分量)
    yaw_rad = math.radians(yaw)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # print(f"turbine rotor plane normal vector: [{cos_yaw}, {sin_yaw}]")

    V_eff = swept_U * cos_yaw + swept_V * sin_yaw  # + swept_W * N_z（如果 N_z ≠ 0）

    # 计算平均有效风速
    average_V_eff = V_eff.mean()

    if debugging_mode:
        return (
            average_V_eff,
            mask.reshape(Z_up, H_up, W_up),
            upsampled_clip_data_tensor,
            clip_data_tensor,
        )
    else:
        # 返回平均风速
        return average_V_eff


def calculate_effective_2d(
    turbine_info: torch.Tensor,
    data_tensor: torch.Tensor,
    debugging_mode: bool = False,
):
    """Calculate the effective U for a single turbine
    Exactly receive the same input as calculate_effective_3d, but only consider the 2D plane at hub height
    1.  Clip the data_tensor around the turbine according to the turbine_info
    2.  Upsample the clip_data_tensor to a higher resolution
    2.5 Clip Again to get 90m horizontal plane
    3.  Traverse all the data point in the clip_data_tensor:
            3.1. if current point is not inside the turbine swept surface, drop it
            3.2. if current point is near the front of turbine swept surface, include it in the mean velocity calculation
    4.  Calculate the velocity component perpendicular to the turbine rotor plane

    Args:
        turbine_info (torch.Tensor): [4, ]: [x_location, y_location, yaw, is_wake_affected]
        data_tensor (torch.Tensor): [C, Z, H, W]
    """

    C, Z, H, W = data_tensor.shape
    x_loc, y_loc, yaw, is_wake_affected = turbine_info.tolist()

    # 根据风机位置裁剪数据，我们只需要考虑风机扫掠面大一点的区域
    # 坐标
    X_from_coord = x_loc - Search_X_Relative
    X_to_coord = x_loc + Search_X_Relative
    Y_from_coord = y_loc - Search_Y_Relative
    Y_to_coord = y_loc + Search_Y_Relative
    Z_from_coord = Search_Z_From - Cell_Size_origin
    Z_to_coord = Search_Z_To - Cell_Size_origin
    # 换算到20m网格的索引
    X_from_idx_origin = int(X_from_coord // Cell_Size_origin)
    X_to_idx_origin = int(X_to_coord // Cell_Size_origin + 1)
    Y_from_idx_origin = int(Y_from_coord // Cell_Size_origin)
    Y_to_idx_origin = int(Y_to_coord // Cell_Size_origin + 1)
    Z_from_idx_origin = int(Z_from_coord // Cell_Size_origin)
    Z_to_idx_origin = int(Z_to_coord // Cell_Size_origin + 1)

    # 截取裁剪后的数据，形状为 [C, Z_clip, H_clip, W_clip]
    clip_data_tensor = data_tensor[
        :, Z_from_idx_origin:Z_to_idx_origin, Y_from_idx_origin:Y_to_idx_origin, X_from_idx_origin:X_to_idx_origin
    ]

    # 对裁剪后的数据进行上采样
    scale_factor = Upsample_scale
    upsampled_clip_data_tensor = upsample_data_3d(clip_data_tensor, scale_factor=scale_factor)
    C, Z_up, H_up, W_up = upsampled_clip_data_tensor.shape

    # 再次裁剪，只保留 90m 高度的水平面
    Z_idx_hub = int((Hub_height - Cell_Size_origin) / Cell_Size_after)
    # print(f"Z_idx_hub: {Z_idx_hub}, Z_up: {Z_up}")
    upsampled_clip_data_tensor = upsampled_clip_data_tensor[:, Z_idx_hub, :, :]
    # upsampled_clip_data_tensor: [C, H_up, W_up]

    # 获取上采样后每个数据点的坐标
    x_coords = torch.linspace(X_from_coord, X_to_coord, W_up)
    y_coords = torch.linspace(Y_from_coord, Y_to_coord, H_up)

    # 创建网格
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # 展平
    x_flat = x_grid.flatten()  # [N]
    y_flat = y_grid.flatten()  # [N]

    # 初始化 z_flat，为了兼容之前的函数形式，这里 z_flat 全部设为风机高度
    z_flat = torch.full_like(x_flat, Hub_height)

    # 计算是否在扫掠面内
    mask = is_point_in_swept_area_direct_batch(
        x_flat, y_flat, z_flat, turbine_info, Sweap_Radius, tolerance=Cell_Size_after * 64
    )

    # 获取在扫掠面内的风速
    swept_U = upsampled_clip_data_tensor[0, :, :].flatten()[mask]
    swept_V = upsampled_clip_data_tensor[1, :, :].flatten()[mask]
    # 二维情况下，没有 W 分量，可以忽略或设为 0
    # swept_W = torch.zeros_like(swept_U)

    # 计算每个点的有效风速分量（平行于风轮法向量的分量）
    yaw_rad = math.radians(yaw)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    V_eff = swept_U * cos_yaw + swept_V * sin_yaw

    # 计算平均有效风速
    average_V_eff = V_eff.mean()

    # 返回平均有效风速
    if debugging_mode:
        return (
            average_V_eff,
            mask.reshape(H_up, W_up),
            upsampled_clip_data_tensor,
            clip_data_tensor,
        )
    else:
        return average_V_eff


# ############################################################# #
#                        test playground                        #
# ############################################################# #


# import torch

# a = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
# b = a * 2 - 8
# print(f"a: {a}")
# print(f"b: {b}")

# mask = b > 0
# print(f"mask: {mask}")
# print(f"shape of mask: {mask.shape}")

# c = a[mask]
# print(f"c: {c}")
# print(f"shape of c: {c.shape}")

# r = torch.sqrt((c - 5.14) ** 2)
# print(f"r: {r}")
# print(f"shape of r: {r.shape}")

# mask_radius = r < 4
# print(f"mask_radius: {mask_radius}")
# print(f"shape of mask_radius: {mask_radius.shape}")


# M = torch.zeros_like(a,dtype=torch.bool)
# M[mask] = mask_radius
# print(f"M: {M}")
# print(f"shape of M: {M.shape}")
