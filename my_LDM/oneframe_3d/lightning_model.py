from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import omegaconf

from my_LDM.oneframe_3d.bicnn_4 import BiCNN


class Oneframe_Area_lightning(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BiCNN()

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(
            params=self.model.parameters(),
            lr=4.5e-6,
            weight_decay=2e-6,
            decoupled_weight_decay=True,
        )
        return optimizer

    def forward(self, input, floris):
        # input: [B, C=36, H, W]
        # floris: [B, C=2, H, W]
        # cat: [B, C=4, H, W]
        # with torch.no_grad():
        #     input = self.abl_recon_func(input).detach()  # [B, C=36, H, W]
        return self.model(torch.cat((input, floris), dim=1))
    
    def forward_(self, x0):
        return self.model(x0)

