from copy import deepcopy
from typing import Optional, Union, Dict, Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only



class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    做了一些注释和改动，注意多gpu时可能报错，尽管已经应用了Implementation detail的issue里全部的修复，并且更新了lightning的标准(boardcast函数被迁移到strategy里了)
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
    """
    def __init__(self, decay: float = 0.999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False


    @staticmethod
    def get_parameters_dict(pl_module: pl.LightningModule):
        """
        我们不想保存鉴别器和lpips的参数，我们只想保存VAE的参数
        Returns state dictionary from pl_module, excluding all keys starting with 'loss.'.
        For example, in pl_module has metrics, you don't want to return their parameters.
        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        full_state_dict = pl_module.state_dict()
        filtered_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('loss.')}
        return filtered_state_dict
        return pl_module.state_dict()
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """ 初始化 ema_state_dict """
        print(f"on_train_start: ema_device: {self.ema_device}")
        # Only keep track of EMA weights in rank zero.
        # torch读入callback的state_dict时，会读到cpu上，这里需要把ema_state_dict放到gpu上
        # if not self._ema_state_dict_ready and pl_module.global_rank == 0:
        if pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_parameters_dict(pl_module))
            if self.ema_device is not None:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()}
                print(f"on_train_start: ema_state_dict device: {self.ema_device}")

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        """ Update EMA weights """
        with torch.no_grad():
            for key, value in self.get_parameters_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """ 大部分是处理多gpu逻辑 单机单卡没有影响; 关键是 pl_module.load_state_dict 加载ema参数 供validation """
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_parameters_dict(pl_module))
        ema_state_dict = pl_module.trainer.strategy.broadcast(self.ema_state_dict, 0)
        self.ema_state_dict = ema_state_dict
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False) #注意这里的strict=False 我们不保存也不想加载鉴别器和lpips参数

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """ Restore original weights"""
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)
    

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """ 大部分是处理多gpu逻辑 单机单卡没有影响; 关键是 pl_module.load_state_dict 加载ema参数 供validation """
        print("on_test_start triggered.xxxxxx")
        print(f"on_test_start: self._ema_state_dict_ready: {self._ema_state_dict_ready}")

        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.
        self.original_state_dict = deepcopy(self.get_parameters_dict(pl_module))
        ema_state_dict = pl_module.trainer.strategy.broadcast(self.ema_state_dict, 0)
        self.ema_state_dict = ema_state_dict
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False) #注意这里的strict=False 我们不保存也不想加载鉴别器和lpips参数
        print("on_test_start triggered.yyyyyy")

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """ Restore original weights"""
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        pl_module.save_hyperparameters()
        trainer.save_checkpoint("last_with_ema.ckpt")
        print("last_with_ema.ckpt saved.")
        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)
    

    # 注意下面这里的state_dick是lightning的callback state，不是torch模型参数的state dict
    # 这个方法是EMACallback的方法，lightning module的state dict方法是直接调用torch的state_dict方法（返回参数）
    def state_dict(self):
        return {
            "ema_state_dict": self.ema_state_dict.copy(),
            "_ema_state_dict_ready": self._ema_state_dict_ready,
        }

    def load_state_dict(self, state_dict):
        print(f"load_state_dict_triggered:{state_dict.keys()} ")
        self.ema_state_dict = state_dict["ema_state_dict"]
        self._ema_state_dict_ready = state_dict["_ema_state_dict_ready"]
        print("ema_state_dict loaded")
        print(f"_ema_state_dict_ready loaded: {self._ema_state_dict_ready}")
        


def load_ema_weight_from_ckpt(ckpt_path: str, model: pl.LightningModule):
    checkpoint = torch.load(ckpt_path)
    ema_state_dict = checkpoint["callbacks"]["EMA"]["ema_state_dict"]
    model.load_state_dict(ema_state_dict, strict=False)
    print("EMA weights loaded from checkpoint.")
    return model