import torch
import torchvision.transforms.functional as TF
import numpy as np
import psutil
from typing import Tuple, Optional, Union, List

from my_LDM.utils.dataset_tool.path_tool import get_root_path

root_path = get_root_path()


# ############################################################# #
#                         normalization                         #
# ############################################################# #


def get_min_max_range():
    """我们使用前7个case的数据(去掉最后40帧)来计算min和max，第8个case用于测试，因此不使用"""

    data_adm_full = np.concatenate(
        [np.load(f"{root_path}/dataset/flowField_full/150_200/adm_{i}.npy")[:-40] for i in range(1, 8)], axis=0
    )
    min = np.min(data_adm_full)
    max = np.max(data_adm_full)
    return min, max


def min_max_normalize_tensor(tensor: torch.Tensor, min: float | torch.Tensor, max: float | torch.Tensor):
    """should to [-1,1]"""
    return (tensor - min) / (max - min) * 2 - 1


def min_max_denormalize_tensor(tensor: torch.Tensor, min: float, max: float):
    """should from [-1,1]"""
    return (tensor + 1) / 2 * (max - min) + min


# ############################################################# #
#                            padding && crop                            #
# ############################################################# #


def pad_crop_adjust(
    img: torch.Tensor,
    to_size: Tuple[int, int],
    random: bool = False,
) -> Tuple[Union[torch.Tensor, np.ndarray], Tuple[int, int]]:
    """根据to_size对img进行padding或者裁剪操作，返回调整后的img。
       使用center crop或random crop处理裁剪操作。

    Args:
        img (torch.Tensor): 输入图像，支持torch.Tensor或np.ndarray。
        to_size (Tuple[int, int]): 目标大小 (H, W)。
        random (bool, optional): 如果为True，使用random crop；否则使用center crop。默认为False。

    Returns:
        Tuple[Union[torch.Tensor, np.ndarray], Tuple[int, int]]: 返回调整后的img和原始大小。
    """
    h, w = img.shape[-2:]
    target_h, target_w = to_size

    # 确保要么是两者都需要 padding，要么是两者都需要 crop，要么是尺寸相等
    assert (
        (h < target_h and w < target_w) or (h > target_h and w > target_w) or (h == target_h and w == target_w)
    ), "Both dimensions must either be padded, cropped, or equal in size."

    # 如果大小匹配，直接返回
    if (h, w) == (target_h, target_w):
        img, original_size = img, (h, w)

    # padding
    if h < target_h or w < target_w:
        img, original_size = pad_to(img, to_size)

    # crop
    if h > target_h or w > target_w:
        img, original_size = crop_to(img, to_size, random=random)

    return img, original_size


def pad_to(img: torch.Tensor, to_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    h, w = img.shape[-2:]
    pad_h = to_size[0] - h
    pad_w = to_size[1] - w

    # 对四周进行对称padding
    if pad_h > 0 or pad_w > 0:
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return img, (h, w)


def crop_to(img: torch.Tensor, to_size: Tuple[int, int], random: bool = False) -> torch.Tensor:
    h, w = img.shape[-2:]
    target_h, target_w = to_size

    if random:
        # 使用随机裁剪
        img = TF.resized_crop(
            img,
            top=random.randint(0, h - target_h),
            left=random.randint(0, w - target_w),
            height=target_h,
            width=target_w,
            size=to_size,
        )
    else:
        # 使用中心裁剪
        img = TF.center_crop(img, to_size)

    return img, (h, w)


def unpad_to(img: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    h, w = original_size
    current_h, current_w = img.shape[-2:]

    # 计算需要移除的padding大小
    pad_h = current_h - h
    pad_w = current_w - w

    # 如果有 padding，则裁剪掉
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 使用切片操作移除 padding，按照 [H, W] 的顺序裁剪
        img = img[..., pad_top : current_h - pad_bottom, pad_left : current_w - pad_right]

    return img


# ############################################################# #
#                           Misc funcs                           #
# ############################################################# #


def check_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
