"""可选依赖兼容层。

原工程的 networks/model.py 与 train.py 依赖 timm / transformers；
darcy_loss.py 依赖 h5py(git 版 mat 文件)。这三个包在部分运行环境里没装,
这里提供等价实现并统一从本模块导入：

    timm          : DropPath, trunc_normal_
    transformers  : PretrainedConfig, get_cosine_schedule_with_warmup

数值行为与所替代的实现一致; 若环境里装了原包则优先使用原包。
"""

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# --------------------------------------------------------------------------
# timm
# --------------------------------------------------------------------------
try:  # 优先使用 timm(若已安装)
    from timm.models.layers import DropPath, trunc_normal_  # noqa: F401

    TIMM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - 依赖缺失时走的路径
    TIMM_AVAILABLE = False

    class DropPath(nn.Module):
        """timm.models.layers.DropPath 的等价实现(stochastic depth, 逐样本丢弃残差分支)。"""

        def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
            super().__init__()
            self.drop_prob = float(drop_prob)
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and self.scale_by_keep:
                random_tensor = random_tensor.div(keep_prob)
            return x * random_tensor

        def extra_repr(self):
            return f"drop_prob={round(self.drop_prob, 3):0.3f}"

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
        """timm.trunc_normal_ 的等价实现。

        注意 timm 的 a/b 是 **以 std 为单位** 的截断界(默认 ±2σ),
        而 torch.nn.init.trunc_normal_ 的 a/b 是绝对数值, 所以要换算。
        """
        return torch.nn.init.trunc_normal_(
            tensor,
            mean=mean,
            std=std,
            a=mean + a * std,
            b=mean + b * std,
            generator=generator,
        )


# --------------------------------------------------------------------------
# transformers
# --------------------------------------------------------------------------
try:  # 优先使用 transformers(若已安装)
    from transformers import (  # noqa: F401
        PretrainedConfig,
        get_cosine_schedule_with_warmup,
    )

    TRANSFORMERS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - 依赖缺失时走的路径
    TRANSFORMERS_AVAILABLE = False

    class PretrainedConfig:
        """transformers.PretrainedConfig 的最小等价实现: 只做属性容器。

        本工程只用它存放 Config 的字段(见 networks/model.py 的 Config 类),
        不涉及 AutoModel / from_pretrained。
        """

        model_type = ""

        def __init__(self, **kwargs):
            self._attn_implementation = kwargs.pop("attn_implementation", "eager")
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        def __repr__(self):
            return f"{self.__class__.__name__}({self.to_dict()})"

    def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
    ):
        """与 transformers.get_cosine_schedule_with_warmup 完全一致的实现。"""
        num_warmup_steps = int(num_warmup_steps)
        num_training_steps = int(num_training_steps)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)
