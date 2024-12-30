import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from typing import Dict, Iterable, Tuple
import numpy as np
import os
import torch.nn.functional as F
import einops

from my_LDM.utils.dataset_tool.path_tool import get_root_path
from my_LDM.utils.dataset_tool.misc import get_min_max_range, min_max_normalize_tensor, check_memory_usage

root_path = get_root_path()
print(f"root_path: {root_path}")


class Inflow_Area_Dataset(Dataset):
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
        min_val=None,
        max_val=None,
    ) -> None:

        super().__init__()

        self.input_data_list = input_data_list
        # [ (T, C=2, Z=12, H=150, W=4) × N ]
        self.floris_data_list = floris_data_list
        # [ (C=2, H=150, W=200) × N ]
        self.target_data_list = target_data_list
        # [ (T, C=2, Z=12, H=150, W=200) × N ]

        self.seq_len = seq_len
        self.max_val = max_val
        self.min_val = min_val


        self.indices = []

        for i, (input_data, floris_data, target_data) in enumerate(
            zip(input_data_list, floris_data_list, target_data_list)
        ):
            """
            # i: 对应case_num 0-12
            # input_data: (Timestep=401, C=2, Z=12, H=150, W=8)
            # floris_data: (C=2, H=150, W=200)
            # target_data: (Timestep=401, C=2, Z=12, H=150, W=200)
            """
            # 这里的timestep是指的数据的时间步，不是预测的时间步
            # 从0开始，因此最后一个时间步是input_data.shape[0] - 1
            for t in range(input_data.shape[0] - self.seq_len):
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
            raise NotImplementedError("min_val and max_val must be provided, at least now")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        
        idxes = self.indices[index]
        input = self.input_data_list[idxes["case_num"]][idxes["input_idx"]]
        floris = self.floris_data_list[idxes["case_num"]]
        target = self.target_data_list[idxes["case_num"]][idxes["target_data_idx"]]

        # Z = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11]
        # Z = [20,40,60,80,100,120,140,160,180,200,220,240]

        if self.min_val is not None:
            input = min_max_normalize_tensor(torch.tensor(input, dtype=torch.float32), self.min_val, self.max_val)
            # 先flip时间步，然后再连接
            # 如果先连接，那么flip时会把原本的W维度也flip，相当于每个时间步细条都是反着的
            input = torch.flip(input, dims=[0])
            input = einops.rearrange(input, "T C Z H W -> (C Z) H (T W)")

            floris = min_max_normalize_tensor(torch.tensor(floris, dtype=torch.float32), self.min_val, self.max_val)

            target = min_max_normalize_tensor(torch.tensor(target, dtype=torch.float32), self.min_val, self.max_val)
            target = einops.rearrange(target, "C Z H W -> (C Z) H W")

        return input, floris, target


class Inflow_Area_DataModule(L.LightningDataModule):
    """
    利用过去50帧入流数据，预测最后一帧的流场状态
    """

    def __init__(
        self,
        data_case_range: Tuple[int, int] = (1, 12),
        target_cases: Iterable[int] = [6, 9, 10, 11, 12],
        valid_isolated_cases: Iterable[int] = [12],
        seq_len: int = 50,
        input_data_dir: str = f"{root_path}/dataset/flowField_measured_area_3d/150_200/",
        floris_data_dir: str = f"{root_path}/dataset/Floris_realCase/150_200/",
        target_data_dir: str = f"{root_path}/dataset/flowField_full_3d/150_200/",
        dataloader_configDict: Dict = {
            # 8 for single-step(22gb video memory is not enough to train single-step) and accumulate 8 steps to get 64 equivalent batch size
            # 16 for rectified flow, accumulate 4 steps to get 64 equivalent batch size
            "batch_size": 8,
            "shuffle": True,
            "num_workers": 8,
            "pin_memory": True,
        },
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.data_case_range = data_case_range
        self.seq_len = seq_len
        self.dataloader_configDict = {
            **{
                "batch_size": 16,
                "shuffle": True,
                "num_workers": 8,
                "pin_memory": True,
            },
            **dataloader_configDict,
        }
        self.normalize = normalize
        # --------------------------- input --------------------------- #
        input_data_list = [
            np.load(os.path.join(input_data_dir, f"adm_{i}.npy"))
            for i in range(data_case_range[0], data_case_range[1] + 1)
        ]
        # -> [ (Timestep=401, C=2, Z=12, H=150, W=200) × 12 ]


        self.input_data_dict_list_train = []
        for i, input_data in enumerate(input_data_list):
            # 下标i+1正好对应case_num
            if i + 1 not in target_cases:
                # 对于非valid集的case，我们把最后的60帧也加入训练
                self.input_data_dict_list_train.append(input_data[:])
            elif i + 1 not in valid_isolated_cases:
                # 对于valid集的case，我们只取前面的数据，后60帧留作valid
                self.input_data_dict_list_train.append(input_data[:-60])
            else:
                # 一定是valid_isolated_cases，不参与训练
                pass
                


        self.input_data_dict_list_valid = []
        for i, input_data in enumerate(input_data_list):
            if i + 1 not in target_cases:
                # 由于非valid集的case所有时间步都加入训练，因此valid集合不加入
                pass
            elif i + 1 not in valid_isolated_cases:
                # 对于留作valid,且一部分数据参与了训练的case, 我们取最后60帧作为valid
                # 由于inflow的迟滞性，我们需要多取seq_len帧，这样valid集的第一个inflow-target对就是正好倒数第60帧
                self.input_data_dict_list_valid.append(input_data[-60 - self.seq_len + 1 :])
            else:
                # 对于与train集完全隔离的case，我们取全部数据作为valid
                self.input_data_dict_list_valid.append(input_data[:])


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
            else:
                pass


        self.floris_data_list_valid = []
        for i, floris_data in enumerate(floris_data_list):
            # 下标i+1正好对应case_num，6 9 10 11是相同风电场不同状态
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
            # 下标i+1正好对应case_num
            if i + 1 not in target_cases:
                # 对于非valid集的case，我们把最后的60帧也带入训练(即所有时间步都参与)
                self.target_data_dict_list_train.append(target_data[:])
            elif i + 1 not in valid_isolated_cases:
                # 对于valid集的case，我们只取前面的数据，后60帧留作valid
                # 如果case_num在valid_isolated_cases中，我们完全隔离于训练，因此全部数据都不参与训练
                self.target_data_dict_list_train.append(target_data[:-60])
            else:
                # 一定是valid_isolated_cases，不参与训练
                pass


        self.target_data_dict_list_valid = []

        for i, target_data in enumerate(target_data_list):
            if i + 1 not in target_cases:
                # 由于非valid集的case所有时间步都加入训练，因此valid集合不加入
                pass
                (f"case {i+1} dropped in valid(target)")
            elif i + 1 not in valid_isolated_cases:
                # 对于留作valid,且一部分数据参与了训练的case, 我们取最后60帧作为valid
                self.target_data_dict_list_valid.append(target_data[-60 - self.seq_len + 1 :])
            else:
                # 对于与train集完全隔离的case，我们取全部数据作为valid
                self.target_data_dict_list_valid.append(target_data[:])


        # ------------------------- normalize ------------------------- #
        if self.normalize:
            self.min_val, self.max_val = get_min_max_range()
            self.min_val = torch.tensor(self.min_val, dtype=torch.float32)
            self.max_val = torch.tensor(self.max_val, dtype=torch.float32)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = Inflow_Area_Dataset(
                input_data_list=self.input_data_dict_list_train,
                floris_data_list=self.floris_data_list_train,
                target_data_list=self.target_data_dict_list_train,
                seq_len=self.seq_len,
                min_val=self.min_val,
                max_val=self.max_val,
            )
            self.val_dataset = Inflow_Area_Dataset(
                input_data_list=self.input_data_dict_list_valid,
                floris_data_list=self.floris_data_list_valid,
                target_data_list=self.target_data_dict_list_valid,
                seq_len=self.seq_len,
                min_val=self.min_val,
                max_val=self.max_val,
            )
        if stage == "test":
            # 我们test和valid使用相同的部分
            self.test_dataset = Inflow_Area_Dataset(
                input_data_list=self.input_data_dict_list_valid,
                floris_data_list=self.floris_data_list_valid,
                target_data_list=self.target_data_dict_list_valid,
                seq_len=self.seq_len,
                min_val=self.min_val,
                max_val=self.max_val,
            )
        if stage == "predict":
            # same as test
            self.test_dataset = Inflow_Area_Dataset(
                input_data_list=self.input_data_dict_list_valid,
                floris_data_list=self.floris_data_list_valid,
                target_data_list=self.target_data_dict_list_valid,
                seq_len=self.seq_len,
                min_val=self.min_val,
                max_val=self.max_val,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_configDict)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **{**self.dataloader_configDict, "shuffle": False})

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **{**self.dataloader_configDict, "shuffle": False})

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **{**self.dataloader_configDict, "shuffle": False})
