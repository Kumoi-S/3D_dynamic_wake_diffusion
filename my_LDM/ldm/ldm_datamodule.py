import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from typing import Dict, Iterable, List, Tuple
import numpy as np
import os
import torch.nn.functional as F
import einops

from my_LDM.utils.dataset_tool.path_tool import get_root_path
from my_LDM.utils.dataset_tool.misc import get_min_max_range, min_max_normalize_tensor, pad_crop_adjust

root_path = get_root_path()
print(f"root_path: {root_path}")


def data_tensor_wrap(tensor: torch.Tensor, filter: slice = None):
    """用于处理多个数据不同slice的情况

    Args:
        tensor (torch.Tensor)
        filter (slice, optional): 数据的使用范围，filter以外的数据不会被数据集使用. Defaults to None.

    Returns:
        dict: "data"访问原始tensor, "filter"访问filter slice
    """
    return {
        "data": tensor,
        "filter": filter if filter is not None else slice(None),
    }


def data_tensor_apply_filter(data_dict: dict):
    filter = data_dict["filter"]
    return data_dict["data"][filter]


class Inflow_Area_Recon_Dataset(Dataset):
    """3d数据集 适用于多个case给定的不同时间步的数据

    Args:
        input_data_dict_list (list[dict]): 传入的是measured data 宽度上被截取
        floris_data_list (list): 每个case对应一个floris data，因此list长度与input和target相同
        target_data_list (list[dict]): 完整流场数据，没有时间步偏移
        seq_len (int):
        min_val (_type_, optional): Defaults to None.
        max_val (_type_, optional): Defaults to None.

    Raises:
        NotImplementedError: 必须传入min和max值，否则会报错
    """

    def __init__(
        self,
        input_data_list: list[dict],
        floris_data_list: list,
        target_data_list: list[dict],
        seq_len: int,
        target_size=None,
        min_val=None,
        max_val=None,
    ) -> None:
        # case_num
        assert (
            len(input_data_list) == len(floris_data_list) == len(target_data_list)
        ), f"case_num don't match: \n input_data_dict_list length: {len(input_data_list)}, floris_data_list length: {len(floris_data_list)}, target_data_list length: {len(target_data_list)}"

        self.input_data_list = input_data_list
        self.floris_data_list = floris_data_list
        self.target_data_list = target_data_list
        self.seq_len = seq_len
        self.target_size = target_size
        self.max_val = max_val
        self.min_val = min_val

        self.indices = []

        for i, (input_data, floris_data, target_data) in enumerate(
            zip(input_data_list, floris_data_list, target_data_list)
        ):
            """
            # i: 对应case_num 0-10
            # input_data: (Timestep=401, C=2, Z=12, H=150, W=8)
            # floris_data: (C=2, H=150, W=200)
            # target_data: (Timestep=401, C=2, Z=12, H=150, W=200)
            """
            input_data_filtered = data_tensor_apply_filter(input_data)
            target_data_filtered = data_tensor_apply_filter(target_data)

            assert (
                input_data_filtered.shape[0] == target_data_filtered.shape[0]
            ), f"subscript{i} of input_data_list and target_data_list must have same timestep"
            assert (
                input_data_filtered.shape[1:-1] == target_data_filtered.shape[1:-1]
            ), f"subscript{i} of input_data_list and target_data_list must have same channel,Z,H"
            assert (
                input_data_filtered.shape[0] >= self.seq_len
            ), f"subscipt{i} of input_data_list must have more timestep than seq_len"

            # 这里的timestep是指的数据的时间步，不是预测的时间步
            # 从0开始，因此最后一个时间步是input_data.shape[0] - 1
            for t in range(input_data_filtered.shape[0] - self.seq_len + 1):
                self.indices.append(
                    {
                        # 由于三个都是list，因此case_num需要单独索引
                        "case_num": i,
                        "input_idx": (slice(t, t + self.seq_len)),
                        "target_data_idx": (t + self.seq_len - 1),
                    }
                )

        if self.min_val is not None and self.max_val is not None:
            self.min_val, self.max_val = get_min_max_range()
            self.min_val = torch.tensor(self.min_val, dtype=torch.float32)
            self.max_val = torch.tensor(self.max_val, dtype=torch.float32)
        else:
            assert min_val is None and max_val is None
            raise NotImplementedError("min_val and max_val must be provided, at least now")

        print(f"len(self.indices): {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index_dict = self.indices[index]
        # [T=50, C=3, Z=12, H=150, W=4]
        input = self.input_data_list[index_dict["case_num"]]["data"][index_dict["input_idx"]]
        # [C=2, H, W]
        floris = self.floris_data_list[index_dict["case_num"]]
        # [C=2, Z=12, H, W]
        target = self.target_data_list[index_dict["case_num"]]["data"][index_dict["target_data_idx"]]
        # Z = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11]
        # Z = [20,40,60,80,100,120,140,160,180,200,220,240]

        if self.min_val is not None:
            input = min_max_normalize_tensor(torch.tensor(input, dtype=torch.float32), self.min_val, self.max_val)
            # 先flip时间步，然后再连接
            # 如果先连接，那么flip时会把原本的W维度也flip，相当于每个时间步细条都是反着的
            input = torch.flip(input, dims=[0])
            input = einops.rearrange(input, "T C Z H W -> (C Z) H (T W)")
            if self.target_size is not None:
                input, original_hw = pad_crop_adjust(input, self.target_size)

            floris = min_max_normalize_tensor(torch.tensor(floris, dtype=torch.float32), self.min_val, self.max_val)
            if self.target_size is not None:
                floris, original_hw = pad_crop_adjust(floris, self.target_size)

            target = min_max_normalize_tensor(torch.tensor(target, dtype=torch.float32), self.min_val, self.max_val)
            target = einops.rearrange(target, "C Z H W -> (C Z) H W")
            if self.target_size is not None:
                target, original_hw = pad_crop_adjust(target, self.target_size)

        # 由于latent的特殊性，我们正常预测y,z通道的风速
        # 我们不预测y,z通道的风速了
        # target = target[0:12]

        return input, floris, target


class Inflow_Area_Recon_DataModule(L.LightningDataModule):
    """
    利用过去50帧入流数据，预测最后一帧的流场状态
    """

    def __init__(
        self,
        data_case_range: list[int, int] = (1, 11),
        target_cases: Iterable[int] = [6, 9, 10, 11, 12],
        valid_isolated_cases: Iterable[int] = [12],

        seq_len: int = 50,
        input_data_dir: str = f"{root_path}/dataset/flowField_measured_area_3d/150_200/",
        floris_data_dir: str = f"{root_path}/dataset/Floris_realCase/150_200/",
        target_data_dir: str = f"{root_path}/dataset/flowField_full_3d/150_200/",
        dataloader_configDict: Dict = {
            "batch_size": 8,
            "shuffle": True,
            "num_workers": 8,
            "pin_memory": True,
        },
        target_size: Tuple[int, int] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.data_case_range = data_case_range
        self.seq_len = seq_len
        self.dataloader_configDict = dataloader_configDict
        self.target_size = target_size
        self.normalize = normalize
        # --------------------------- input --------------------------- #
        input_data_list = [
            np.load(os.path.join(input_data_dir, f"adm_{i}.npy"))
            for i in range(data_case_range[0], data_case_range[1] + 1)
        ]
        # -> [ (Timestep=401, C=2, Z=12, H=150, W=200) × 12 ]

        self.input_data_dict_list_train = []
        for i, input_data in enumerate(input_data_list):
            # 下标i+1正好对应case_num，6 9 10 11是相同风电场不同状态
            if i + 1 in target_cases:
                if i + 1 not in valid_isolated_cases:
                    self.input_data_dict_list_train.append(data_tensor_wrap(input_data, filter=slice(None, -60)))
                # case 11我们让他完全不涉入train，仅供valid
            else:
                self.input_data_dict_list_train.append(data_tensor_wrap(input_data, filter=None))
        print(f"Train input_data_dict_list length: {len(self.input_data_dict_list_train)}")

        self.input_data_dict_list_valid = []
        for i, input_data in enumerate(input_data_list):
            # valid集合只包含target_cases
            if i + 1 in target_cases:
                self.input_data_dict_list_valid.append(data_tensor_wrap(input_data, filter=slice(-60, None)))
            else:
                # 其他case我们不关心，全部拿来训练，因此valid集合不加入
                pass
        print(f"Valid input_data_dict_list length: {len(self.input_data_dict_list_valid)}")

        # --------------------------- floris -------------------------- #
        floris_data_list = [
            np.load(os.path.join(floris_data_dir, f"adm_{i}.npy"))
            for i in range(data_case_range[0], data_case_range[1] + 1)
        ]
        # -> [ (C=2, H=150, W=200) × 12 ]

        self.floris_data_list_train = []
        for i, floris_data in enumerate(floris_data_list):
            # 对于floris，无需考虑时间步，只需考虑隔离在valid集的case
            if i + 1 not in valid_isolated_cases:
                self.floris_data_list_train.append(floris_data)
        print(f"Train floris_data_list length: {len(self.floris_data_list_train)}")

        self.floris_data_list_valid = []
        for i, floris_data in enumerate(floris_data_list):
            # valid集合只包含target_cases
            if i + 1 in target_cases:
                self.floris_data_list_valid.append(floris_data)
            else:
                # 其他case我们不关心，全部拿来训练，因此valid集合不加入
                pass

        # --------------------------- target -------------------------- #
        target_data_list = [
            np.load(os.path.join(target_data_dir, f"adm_{i}.npy"))
            for i in range(data_case_range[0], data_case_range[1] + 1)
        ]
        # -> [ (Timestep=401, C=2, Z=12, H=150, W=200) × 12 ]

        self.target_data_dict_list_train = []
        for i, target_data in enumerate(target_data_list):
            # 下标i+1正好对应case_num，6 9 10 11是相同风电场不同状态
            if i + 1 in target_cases:
                if i + 1 not in valid_isolated_cases:
                    self.target_data_dict_list_train.append(data_tensor_wrap(target_data, filter=slice(None, -60)))
                # case 11我们让他完全不涉入train，仅供valid
            else:
                self.target_data_dict_list_train.append(data_tensor_wrap(target_data, filter=None))
        print(f"Train target_data_dict_list length: {len(self.target_data_dict_list_train)}")

        self.target_data_dict_list_valid = []
        for i, target_data in enumerate(target_data_list):
            # 下标i+1正好对应case_num，6 9 10 11是相同风电场不同状态
            if i + 1 in target_cases:
                self.target_data_dict_list_valid.append(data_tensor_wrap(target_data, filter=slice(-60, None)))
            else:
                # 其他case我们不关心，全部拿来训练，因此valid集合不加入
                pass
        print(f"Valid target_data_dict_list length: {len(self.target_data_dict_list_valid)}")

        # ------------------------- normalize ------------------------- #
        if self.normalize:
            self.min_val, self.max_val = get_min_max_range()
            self.min_val = torch.tensor(self.min_val, dtype=torch.float32)
            self.max_val = torch.tensor(self.max_val, dtype=torch.float32)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = Inflow_Area_Recon_Dataset(
                input_data_list=self.input_data_dict_list_train,
                floris_data_list=self.floris_data_list_train,
                target_data_list=self.target_data_dict_list_train,
                seq_len=self.seq_len,
                target_size=self.target_size,
                min_val=self.min_val,
                max_val=self.max_val,
            )
            self.val_dataset = Inflow_Area_Recon_Dataset(
                input_data_list=self.input_data_dict_list_valid,
                floris_data_list=self.floris_data_list_valid,
                target_data_list=self.target_data_dict_list_valid,
                seq_len=self.seq_len,
                target_size=self.target_size,
                min_val=self.min_val,
                max_val=self.max_val,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_configDict)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **{**self.dataloader_configDict, "shuffle": False})
