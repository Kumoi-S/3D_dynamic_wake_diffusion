from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import omegaconf
import time
import einops

# from my_LDM.rectified_flow.bicnn import BiCNN
# from my_LDM.rectified_flow.bicnn_2 import BiCNN
# from my_LDM.rectified_flow.bicnn_3 import BiCNN
from my_LDM.rectified_flow.bicnn_4 import BiCNN
# from my_LDM.rectified_flow.bicnn_5 import BiCNN
# from my_LDM.rectified_flow.unet import BiCNN
# from my_LDM.rectified_flow.unet_2 import BiCNN
from my_LDM.utils.dataset_tool.misc import min_max_normalize_tensor, min_max_denormalize_tensor, get_min_max_range

# 计算 R^2 函数（逐batch维度计算）
def r2_score_batch(pred, target):
    """Calculate R^2 score for each sample in a batch.
    
    Args:
        pred: [B, ...]
        target: [B, ...]
    """
    batch_r2_scores = []
    for i in range(pred.size(0)):  # 遍历 batch 维度
        ss_total = torch.sum((target[i] - target[i].mean()) ** 2)
        ss_residual = torch.sum((target[i] - pred[i]) ** 2)
        r2 = 1 - ss_residual / ss_total if ss_total != 0 else torch.tensor(0.0, device=pred.device)
        batch_r2_scores.append(r2)
    return torch.mean(torch.stack(batch_r2_scores))


class Rectified_flow(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = BiCNN()
        self.N = 10
        # 仅用于test
        self.data_min = None
        self.data_max = None
        # 用于记录batch开始时间
        self.batch_start_time = None

    def configure_optimizers(self):
        return torch.optim.NAdam(
            params=self.net.parameters(),
            lr=4.5e-6,  
            weight_decay=2e-6, 
            decoupled_weight_decay=True,
        )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        if self.data_min is None:
            self.data_min, self.data_max = get_min_max_range()
            self.data_min, self.data_max = (
                torch.tensor(self.data_min).to(self.device),
                torch.tensor(self.data_max).to(self.device),
            )
            print("self.data_min:", self.data_min)
            print("type of self.data_min:", type(self.data_min))
            print("self.data_max:", self.data_max)
            print("type of self.data_max:", type(self.data_max))

        x0, y, x1 = batch
        # with torch.no_grad():
        #     x0 = self.abl_recon_func(x0).detach()

        B, C, H, W = x1.shape

        x0 = torch.cat([x0, y], dim=1)
        x1 = torch.cat([x1, y], dim=1)



        for N in [1, 2, 3, 5, 10, 30, 50]:
            pred_traj = self.sample_ode(x0, N)
            pred = pred_traj[-1]
            target = x1

            # 逆归一化
            pred = min_max_denormalize_tensor(pred, self.data_min, self.data_max)
            target = min_max_denormalize_tensor(target, self.data_min, self.data_max)

            # 我们不考虑floris，只针对原始数据计算valid loss
            loss_mse = F.mse_loss(input=pred[:, :C], target=target[:, :C])
            loss_l1 = F.l1_loss(input=pred[:, :C], target=target[:, :C])
            loss_rmse = torch.sqrt(loss_mse)
            loss_r2 = r2_score_batch(pred[:, :C], target[:, :C])

            self.log(f"Test/real_MSE_{N}", loss_mse, prog_bar=True)
            self.log(f"Test/real_L1_{N}", loss_l1, prog_bar=True)
            self.log(f"Test/real_RMSE_{N}", loss_rmse, prog_bar=True)
            self.log(f"Test/real_R2_{N}", loss_r2, prog_bar=True)

            # 查看所有batch, 因此不限定batch_idx
            for i, (pred_view, target_view) in enumerate(zip(pred, target)):
                fig = self.plot_channels_for_test(pred_view, target_view, timestep=0)
                self.logger.experiment.add_figure(f"Test/batch{batch_idx} {N}step", fig, global_step=self.global_step)
                # 我们只看batch中的第一个样本
                break

    def sample_forward(self, x0, x1):
        eps = x1 - x0  # 这是我们预测的
        t = torch.rand((x1.shape[0],), device=self.device)

        _t = t.reshape(-1, 1, 1, 1)
        xt = _t * x1 + (1.0 - _t) * x0  # 等价于 xt = x0 + t * eps
        return xt, t, eps

    def sample_ode(self, x0=None, N=None):
        """Use Euler method to sample from the learned flow
        Returns:
            traj: list of tensors, [x0, x1, x2, ..., xN], first is the initial state x0, last is the final state xN
        """
        if N is None:
            N = self.N
        dt = 1.0 / N

        traj = []  # to store the trajectory
        x = x0.detach().clone()
        B = x.shape[0]

        traj.append(x.detach().clone())

        for i in range(N):
            t = torch.ones((B,), device=self.device) * i / N
            pred = self.net(x, t)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def predict_step(self, batch):
        x0, y, x1 = batch
        # with torch.no_grad():
        #     x0 = self.abl_recon_func(x0).detach()

        B, C, H, W = x1.shape

        x0 = torch.cat([x0, y], dim=1)
        x1 = torch.cat([x1, y], dim=1)

        N = 10
        pred_traj = self.sample_ode(x0, N)
        pred = pred_traj[-1]

        return pred


        # x0, y, x1 = batch
        # x0 = torch.cat([x0, y], dim=1)
        # return self.sample_ode(x0, N=2)

    
    # 最佳设置
    def sample_ode_fast(self, x0=None):
        """the same as above but 1)Assume N=2 2)dont record trajaectory"""
        # dt = 0.5

        # x = x0
        # B = x.shape[0]

        # with torch.no_grad():
        #     for i in range(2):
        #         t = torch.ones((B,), device=self.device) * i / 2
        #         pred = self.net(x, t)
        #         x = x + pred * dt
        list = self.sample_ode(x0, 2)

        return list[-1]
    

    def forward(self, x0: torch.Tensor, y):
        with torch.no_grad():
            # x0 = self.abl_recon_func(x0).detach()
            x0 = torch.cat([x0, y], dim=1)
            x1 = self.sample_ode(x0, 10)[-1]

        # # 用于torchinfo的测试
        # x0 = torch.cat([x0, y], dim=1)
        # t = torch.rand((x0.shape[0],), device=self.device)
        # x1 = self.net(x0, t)
        return x1

    @staticmethod
    def plot_channels(tensor_pred: torch.Tensor, tensor_target: torch.Tensor, timestep: int) -> plt.Figure:
        """tensor: [C=36, Z, H, W]

        Z = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11]
        Z = [20,40,60,80,100,120,140,160,180,200,220,240]
        tensor_xxxx[Z, 12*C]   C=0,1,2    这样分别访问x y z通道
        """
        tensor_pred = tensor_pred.detach().cpu().numpy()
        tensor_target = tensor_target.detach().cpu().numpy()

        # 6行: 3个通道的样本对; 6列: 6个高度
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        # 对应高度：[40, 80, 100, 140, 180, 220]
        selected_Zs = [1, 3, 4, 6, 8, 10]
        for i, z in enumerate(selected_Zs):
            Z = 20 + 20 * z
            # x
            im1 = axes[0, i].imshow(tensor_pred[z + 12 * 0], cmap="coolwarm", vmin=-1, vmax=1)
            axes[0, i].set_title(f"Pred X Z={Z}, t={timestep}")
            im2 = axes[1, i].imshow(tensor_target[z + 12 * 0], cmap="coolwarm", vmin=-1, vmax=1)
            axes[1, i].set_title(f"Target X Z={Z}, t={timestep}")
            # y
            im3 = axes[2, i].imshow(tensor_pred[z + 12 * 1], cmap="coolwarm", vmin=-1, vmax=1)
            axes[2, i].set_title(f"Pred Y Z={Z}, t={timestep}")
            im4 = axes[3, i].imshow(tensor_target[z + 12 * 1], cmap="coolwarm", vmin=-1, vmax=1)
            axes[3, i].set_title(f"Target Z={Z}, t={timestep}")
            # z
            im5 = axes[4, i].imshow(tensor_pred[z + 12 * 2], cmap="coolwarm", vmin=-1, vmax=1)
            axes[4, i].set_title(f"Pred Z Z={Z}, t={timestep}")
            im6 = axes[5, i].imshow(tensor_target[z + 12 * 2], cmap="coolwarm", vmin=-1, vmax=1)
            axes[5, i].set_title(f"Target Z Z={Z}, t={timestep}")

        # 调整图像间距，给 colorbar 预留空间
        fig.subplots_adjust(right=0.9)
        # 创建共享的 colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax)

        return fig

    @staticmethod
    def plot_channels_for_test(tensor_pred: torch.Tensor, tensor_target: torch.Tensor, timestep: int) -> plt.Figure:
        """tensor: [C=36, H, W]

        Z = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11]
        Z = [20,40,60,80,100,120,140,160,180,200,220,240]
        tensor_xxxx[Z, 12*C]   C=0,1,2    这样分别访问x y z通道
        """
        tensor_pred = tensor_pred.detach().cpu().numpy()
        tensor_target = tensor_target.detach().cpu().numpy()

        # 6行: 3个通道的样本对; 6列: 6个高度
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        # 对应高度：[40, 80, 100, 140, 180, 220]
        selected_Zs = [1, 3, 4, 6, 8, 10]
        for i, z in enumerate(selected_Zs):
            Z = 20 + 20 * z
            # x
            im1 = axes[0, i].imshow(tensor_pred[z + 12 * 0], cmap="coolwarm", vmin=0, vmax=14)
            axes[0, i].set_title(f"Pred X Z={Z}, t={timestep}")
            im2 = axes[1, i].imshow(tensor_target[z + 12 * 0], cmap="coolwarm", vmin=0, vmax=14)
            axes[1, i].set_title(f"Target X Z={Z}, t={timestep}")
            # y
            im3 = axes[2, i].imshow(tensor_pred[z + 12 * 1], cmap="coolwarm", vmin=0, vmax=14)
            axes[2, i].set_title(f"Pred Y Z={Z}, t={timestep}")
            im4 = axes[3, i].imshow(tensor_target[z + 12 * 1], cmap="coolwarm", vmin=0, vmax=14)
            axes[3, i].set_title(f"Target Z={Z}, t={timestep}")
            # z
            im5 = axes[4, i].imshow(tensor_pred[z + 12 * 2], cmap="coolwarm", vmin=0, vmax=14)
            axes[4, i].set_title(f"Pred Z Z={Z}, t={timestep}")
            im6 = axes[5, i].imshow(tensor_target[z + 12 * 2], cmap="coolwarm", vmin=0, vmax=14)
            axes[5, i].set_title(f"Target Z Z={Z}, t={timestep}")

        # 调整图像间距，给 colorbar 预留空间
        fig.subplots_adjust(right=0.9)
        # 创建共享的 colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax)

        return fig

    
    @staticmethod
    def plot_channels_for_test_diffch(
        tensor_pred: torch.Tensor,
        tensor_target: torch.Tensor,
        channel: int,
        channel_name="X",
        vmin=-3,
        vmax=11,
        pred_title="Pred",
        target_title="Target",
    ) -> plt.Figure:
        """
        Plot the prediction, target, and difference of a given channel from 3D tensors.

        tensor: [C=3, Z=12, H, W]

        Z = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11]
        Z = [20,40,60,80,100,120,140,160,180,200,220,240]
        """
        tensor_pred = tensor_pred.detach().cpu().numpy()
        tensor_target = tensor_target.detach().cpu().numpy()
        tensor_diff = np.abs(tensor_pred - tensor_target)

        # 3 rows: prediction, target, difference; 6 columns: 6 heights
        fig, axes = plt.subplots(3, 6, figsize=(18, 8))
        selected_Zs = [1, 3, 4, 6, 8, 10]
        for i, z in enumerate(selected_Zs):
            Z = 20 + 20 * z

            # Plot prediction
            im1 = axes[0, i].imshow(tensor_pred[channel, z], cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"{pred_title}: {channel_name} Z={Z}")

            # Plot target
            im2 = axes[1, i].imshow(tensor_target[channel, z], cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"{target_title}: {channel_name} Z={Z}")

            # Plot difference
            im3 = axes[2, i].imshow(tensor_diff[channel, z], cmap="Reds", vmin=0, vmax=vmax)
            axes[2, i].set_title(f"Diff: {channel_name} Z={Z}")

        # Adjust subplot layout to make room for the colorbar
        fig.subplots_adjust(right=0.9)

        # Add a colorbar for prediction and target
        cbar_ax1 = fig.add_axes([0.92, 0.4, 0.02, 0.45])
        fig.colorbar(im1, cax=cbar_ax1, label=f"{channel_name} Value")

        # Add a colorbar for the difference
        cbar_ax2 = fig.add_axes([0.92, 0.12, 0.02, 0.2])
        fig.colorbar(im3, cax=cbar_ax2, label="Difference")

        return fig