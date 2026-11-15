#MIT License
#
#Copyright (c) 2020 Zongyi Li
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import torch
import numpy as np
import scipy.io
from pathlib import Path
import torch.nn as nn


#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            # 只有 MATLAB v7.3(HDF5) 的 .mat 文件才需要 h5py，按需导入，
            # 这样常规训练/推理不再强制依赖 h5py。
            try:
                import h5py
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    f"{self.file_path} 需要 h5py 才能读取(MATLAB v7.3/HDF5); "
                    "请先 python -m pip install h5py，或改用 scipy.io 可读的 .mat 文件"
                ) from exc
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x.float()

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x.float()

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class RmseLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(RmseLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        n_norms = torch.tensor(3339, dtype=torch.float32)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/n_norms)
            else:
                return torch.sum(diff_norms/n_norms)

        return diff_norms/n_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss_weighted(object):
    """深度加权 **相对 L1 / L2** loss；可选把 mask(SSE 区) 内的权重再乘 ``mask_boost`` 倍。

        w_eff = w_depth * (1 + (mask_boost - 1) * mask)

        loss_mode="l1"（推荐，重尾/突变目标）:
            L = Σ w_eff·|x-y|  /  (Σ w_eff·|y| + eps)
        loss_mode="l2"（原实现）:
            L = sqrt(Σ w_eff·(x-y)²) / sqrt(Σ w_eff·y²)

    L1 对大幅度误差的惩罚是线性的、不会像 L2 那样把预测拉向均值，更适合"要有突变"的目标；
    代价是损失值量级与 L2 不同（通常更小），历史曲线不可直接比较，梯度也变成符号型（幅值恒定），
    必要时把 lr 调小一点。

    - ``mask_boost = 1.0``（默认）→ 与改造前逐位一致；
    - ``mask_boost = 5.0`` → mask 区内每个单元的权重 ×5（mask 外的深度加权不变）。

    mask 文件为 ``.npz``，key 默认取区域名（cascadia / alaska / nankai / newzealand），
    值为长度 = 断层单元数的 bool/0-1 数组。
    """

    KNOWN_CASES = ('cascadia', 'alaska', 'newzealand', 'nankai')

    def __init__(self, fault_file, mask_path=None, mask_key=None, mask_boost=1.0, loss_mode="l2"):
        super(LpLoss_weighted, self).__init__()
        self.fault_file = str(fault_file)
        self.case = self._parse_case(self.fault_file)
        self.llh = np.loadtxt(self.fault_file)
        self.depth = -self.llh[:, -1]

        # 深度权重是常量：只在 CPU 算一次，按 device 缓存，避免每个 batch 重算
        self._depth_weight = self._compute_weight(self.depth)
        self._weight_cache = {}

        self.mask_boost = float(mask_boost)
        self.loss_mode = str(loss_mode).lower()
        if self.loss_mode not in ("l1", "l2"):
            raise ValueError(f"loss_mode 只能是 'l1' 或 'l2'，收到 {loss_mode!r}")
        self.mask = None
        self.mask_key = None
        if mask_path and self.mask_boost != 1.0:
            self._load_mask(mask_path, mask_key)

        # 最近一次 __call__ 的分量（日志用）
        self.last_components = {}

    # ---------------------------------------------------------------- 区域名
    @classmethod
    def _parse_case(cls, fault_file):
        """从路径里认区域名。

        原实现用 ``fault_file.split('/')[2]``，只对 ``./Data/cascadia/fault/llh``
        这种相对路径成立；换成绝对路径会取到 'project' 导致 UnboundLocalError。
        """
        parts = [p.lower() for p in Path(str(fault_file)).parts]
        for part in parts:
            if part in cls.KNOWN_CASES:
                return part
        raise ValueError(
            f"无法从 fault_file 路径中识别区域名(可选 {cls.KNOWN_CASES}): {fault_file}"
        )

    # ---------------------------------------------------------------- 权重
    def _compute_weight(self, depth):
        #不同的case的权重
        if self.case in ('cascadia', 'alaska'):
            weight = torch.tensor(np.exp(-((depth - 35) ** 2) / (2 * 20 ** 2)))
        elif self.case == 'newzealand':
            weight = torch.tensor(np.exp(-((depth - 10) ** 2) / (2 * 7.5 ** 2)))
        elif self.case == 'nankai':
            weight = torch.tensor(np.exp(-((depth - 45) ** 2) / (2 * 20 ** 2)))
        else:  # pragma: no cover - _parse_case 已经保证不会走到这里
            raise ValueError(f"未定义的 case: {self.case}")

        return weight.float()

    def _load_mask(self, mask_path, mask_key):
        key = mask_key or self.case
        with np.load(str(mask_path), allow_pickle=False) as z:
            if key not in z:
                raise KeyError(
                    f"mask 文件 {mask_path} 里没有 key='{key}'，可用: {list(z.keys())}"
                )
            m = np.asarray(z[key])
        if m.shape[0] != self._depth_weight.numel():
            raise ValueError(
                f"mask 与断层单元数不一致: mask {m.shape[0]} vs fault {self._depth_weight.numel()}"
            )
        self.mask = torch.as_tensor(m.astype(np.float32))
        self.mask_key = key
        # 把 boost 折进权重矩阵，之后每次算 loss 只是一次乘法
        factor = 1.0 + (self.mask_boost - 1.0) * self.mask
        self._depth_weight = self._depth_weight * factor
        self._weight_cache = {}
        print(f"[LpLoss_weighted] loss_mode={self.loss_mode} mask='{key}' {int(self.mask.sum())}/{self.mask.numel()} "
              f"({100 * float(self.mask.mean()):.1f}%) | 权重 ×{self.mask_boost:g} "
              f"(mask 区平均权重 {float(self._depth_weight[self.mask > 0].mean()):.3f}, "
              f"非 mask 区 {float(self._depth_weight[self.mask == 0].mean()):.3f})")

    def weight_on(self, device):
        tensor = self._weight_cache.get(device)
        if tensor is None:
            tensor = self._depth_weight.to(device)
            self._weight_cache[device] = tensor
        return tensor

    def mask_on(self, device):
        """返回 mask 张量（未启用 mask 时为 None）。"""
        return None if self.mask is None else self.mask.to(device)

    # ---------------------------------------------------------------- 计算
    def rel(self, x, y, weight=None, norm=None):
        """加权相对误差；``weight`` 为 None 时用（可能含 mask 加成的）深度权重。

        ``self.loss_mode == "l1"`` → Σ w|x-y| / Σ w|y|；``"l2"`` → 原来的 √(Σ w(x-y)²)/√(Σ wy²)。

        ``norm``：可选的**分母来源**（与 ``y`` 同形状）。给定时分母取 ``norm`` 而不是 ``y``，
        用于"误差很小、但量纲上要相对某个大尺度量归一"的情形 —— 典型用法是预测增量 Δ：
            L_Δ = Σ w|Δp-Δt| / Σ w|y_t|        （分母是绝对场幅度，不是 Δ 本身）
        否则 Δ 接近 0 时 Σw|Δt| 趋近 0，相对误差会爆掉（梯度变噪声）。
        """
        if weight is None:
            weight = self.weight_on(x.device)
        den = y if norm is None else norm
        if self.loss_mode == "l1":
            diff_norms = torch.sum(weight * (x - y).abs(), axis=(1, 2))   # 加权 L1
            y_norms = torch.sum(weight * den.abs(), axis=(1, 2))          # 分母的加权 L1
        else:
            diff_norms = torch.sqrt(torch.sum(weight * (x - y) ** 2, axis=(1, 2)))  # 对后两维计算L2范数
            y_norms = torch.sqrt(torch.sum(weight * den ** 2, axis=(1, 2)))         # 分母的L2范数
        loss = torch.mean(diff_norms / (y_norms + 1e-6))
        return loss

    def __call__(self, x, y, norm=None):
        loss = self.rel(x, y, norm=norm)
        self.last_components = {'all': float(loss.detach())}
        return loss


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
