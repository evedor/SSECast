import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import os


def get_dataloader(args, purpose):
    dataset = GetDataset(args, purpose)

    # 获取num_workers配置，默认为4
    num_workers = getattr(args, 'num_workers', 4)

    if purpose == 'train':
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                pin_memory=True, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, num_workers=num_workers)

    return dataloader


class GetDataset(Dataset):
    """滑动窗口数据集。

    ``target_mode="next_day"``（默认）:
        x = 连续 ``time_chans`` 天, y = 其后 ``rollout_horizon`` 天（逐日）
    ``target_mode="multi_window"``:
        y[k] = ``[t+1+k, t+1+k+target_window)`` 这 ``target_window`` 天的平均场，
        k = 0..rollout_horizon-1。例如 target_window=5, rollout_horizon=18 →
        依次是 1-5 天平均、2-6 天平均、…、18-22 天平均（滑动、重叠的 5 天窗）。

    x: (time_chans, data_chans, fea_len)；y: (rollout_horizon, data_chans, fea_len)
    """

    def __init__(self, args, purpose):
        self.args = args
        self.purpose = purpose
        self.data_path = args.data_path
        self.dt = int(getattr(args, 'dt', 1))
        self.use_tremor = args.use_tremor

        self.window = int(getattr(args, 'time_chans', 2))        # 输入天数
        self.horizon = int(getattr(args, 'rollout_horizon', 2))  # 预测总天数
        self.gap = max(0, self.dt - 1)                           # 输入窗与标签之间的间隔(天)
        self.target_mode = str(getattr(args, 'target_mode', 'next_day'))
        self.target_window = int(getattr(args, 'target_window', 1))
        if self.target_mode not in ('next_day', 'multi_window'):
            raise ValueError(f"target_mode 只能是 'next_day' 或 'multi_window'，收到 {self.target_mode!r}")
        if self.target_mode == 'multi_window':
            # 需要预测到 t + horizon + target_window - 1 天
            self.span = self.horizon + self.target_window - 1
        else:
            self.span = self.horizon

        self.dir = os.path.join(args.data_path, purpose)
        self.slip = np.loadtxt(self.dir + '/slip')
        self.slip_strike = np.loadtxt(self.dir + '/slip_strike')
        self.slip_dip = np.loadtxt(self.dir + '/slip_dip')

        self.mean = np.load(args.mean_path)
        self.std = np.load(args.std_path)

        if self.use_tremor:
            tremor, row_index = self._load_tremor(os.path.join(args.tremor_path, purpose), purpose)
            # 对齐到 slip 的行：缺失的日期补 0（新数据集用 slip_row_index.npy 精确对齐，
            # 旧数据集用文本文件 + "取最后 N 天/等长"的老规则）
            aligned = np.zeros((self.slip.shape[0], tremor.shape[1]), dtype=np.float64)
            aligned[row_index] = tremor
            self.tremor_density = aligned
            self.slip_combined = np.stack([self.slip, self.slip_strike, self.slip_dip, aligned], axis=1)
        else:
            self.slip_combined = np.stack([self.slip, self.slip_strike, self.slip_dip], axis=1)

        n_ch = self.slip_combined.shape[1]
        self.mean = self._fit_norm(self.mean, n_ch, 0.0)
        self.std = self._fit_norm(self.std, n_ch, 1.0)
        self.std = np.where(self.std == 0, 1.0, self.std)      # 防止除 0（旧数据里 tremor 通道 std=0）
        self.slip_combined = (self.slip_combined - self.mean[None, :, None]) / (self.std[None, :, None])
        desc = (f'{self.horizon} 个 {self.target_window} 天滑窗（{1}-{self.target_window} … '
                f'{self.horizon}-{self.horizon + self.target_window - 1} 天平均）'
                if self.target_mode == 'multi_window' else f'预测 {self.horizon} 天')
        print(purpose + '数据初始化完成', '| 窗口', self.window, '天 ->', desc, '| 样本数', len(self))


    @staticmethod
    def _fit_norm(values, n_ch, fill):
        """把 mean/std 补齐到通道数：多了截断，少了用 fill 补（默认 mean=0 / std=1）。"""
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size >= n_ch:
            return values[:n_ch]
        return np.concatenate([values, np.full(n_ch - values.size, fill, dtype=np.float64)])

    def _load_tremor(self, tremor_dir, purpose):
        """读 tremor_density，并给出它在 slip 表里的行号。

        支持两种布局：
          (a) 新数据集：``tremor_density.npy`` (+ ``slip_row_index.npy`` / ``dates.npy``)
          (b) 旧数据集：文本文件 ``tremor_density``（train 取 slip 的最后 N 行，eval/test 等长）
        """
        npy = os.path.join(tremor_dir, 'tremor_density.npy')
        idx = os.path.join(tremor_dir, 'slip_row_index.npy')
        n_slip = self.slip.shape[0]
        if os.path.isfile(npy) and os.path.isfile(idx):
            density = np.load(npy).astype(np.float64)
            row_index = np.load(idx).astype(np.int64)
            if density.shape[0] != row_index.size:
                raise ValueError(f'{npy} 行数 {density.shape[0]} 与 slip_row_index {row_index.size} 不一致')
            if row_index.min() < 0 or row_index.max() >= n_slip:
                raise ValueError(f'{idx} 行号越界: [{row_index.min()}, {row_index.max()}] vs slip {n_slip} 行')
            return density, row_index

        density = np.loadtxt(os.path.join(tremor_dir, 'tremor_density'))
        density = np.atleast_2d(density)
        if density.shape[0] == n_slip:
            row_index = np.arange(n_slip)
        elif purpose == 'train' and density.shape[0] < n_slip:
            row_index = np.arange(n_slip - density.shape[0], n_slip)
        else:
            raise ValueError(
                f'tremor 行数 {density.shape[0]} 与 slip {n_slip} 不一致（{tremor_dir}）'
            )
        return density, row_index

    def __len__(self):
        return self.slip_combined.shape[0] - self.window - self.gap - self.span + 1

    def __getitem__(self, index):
        label_start = index + self.window + self.gap
        x = self.slip_combined[index:index + self.window]
        if self.target_mode == 'multi_window':
            # y[k] = 从 label_start+k 起连续 target_window 天的平均
            w = self.target_window
            y = np.stack([self.slip_combined[label_start + k: label_start + k + w].mean(axis=0)
                          for k in range(self.horizon)], axis=0)
        else:
            y = self.slip_combined[label_start:label_start + self.horizon]
        return (torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32))


# ====================================================================== 多区域
class MultiRegionDataset(Dataset):
    """把多个区域的 ``GetDataset`` 拼起来，样本为 ``(x, y, region_idx)``。

    每个区域用自己的 ``data_path / mean_path / std_path``（归一化按区域独立，避免
    slip 幅度差一个量级时互相污染），样本索引全局连续。
    """

    def __init__(self, args, purpose, regions=None, verbose=True):
        self.regions = list(regions if regions is not None else getattr(args, 'regions', []) or [])
        if not self.regions:
            raise ValueError('MultiRegionDataset 需要 regions（或 yaml 里的 regions 列表）')
        from utils.region_registry import region_data_paths
        self.sets, self.offsets = [], [0]
        for r in self.regions:
            a = _RegionArgs(args)
            paths = region_data_paths(args, r)      # 全部来自 yaml 的 region_configs.<region>
            a.data_path = paths['data_path']
            a.mean_path = paths['mean_path']
            a.std_path = paths['std_path']
            a.tremor_path = paths['tremor_path']
            self.sets.append(GetDataset(a, purpose))
            self.offsets.append(self.offsets[-1] + len(self.sets[-1]))
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        if verbose:
            print('[multiregion] %s | %s | 合计 %d 样本'
                  % (purpose, {r: len(s) for r, s in zip(self.regions, self.sets)}, len(self)))

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        rid = int(np.searchsorted(self.offsets, index, side='right') - 1)
        x, y = self.sets[rid][index - int(self.offsets[rid])]
        return x, y, rid


class _RegionArgs:
    """从原 args 复制一份、只改路径的轻量代理（GetDataset 只读属性）。"""

    _FIELDS = ('dt', 'time_chans', 'rollout_horizon', 'target_mode', 'target_window',
               'use_tremor', 'batch_size', 'num_workers')

    def __init__(self, args):
        for f in self._FIELDS:
            if hasattr(args, f):
                setattr(self, f, getattr(args, f))
        self.use_tremor = bool(getattr(args, 'use_tremor', False))
        self.target_mode = str(getattr(args, 'target_mode', 'next_day'))
        self.target_window = int(getattr(args, 'target_window', 1))
        self.num_workers = int(getattr(args, 'num_workers', 4))


class RegionBatchSampler(Sampler):
    """一个 batch 只放**同一区域**的样本（各区域 token 数不同，长度一致就免 padding/mask）。

    每个区域内部打乱后按 batch_size 切块，再把所有区域的 batch 混起来打乱 →
    同一 epoch 内各区域轮转，但单个 batch 长度固定。
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.offsets = np.asarray(dataset.offsets, dtype=np.int64)   # 可被子区域 loader 覆盖

    def __len__(self):
        tot = 0
        for lo, hi in zip(self.offsets[:-1], self.offsets[1:]):
            n = int(hi - lo)
            tot += (n + self.batch_size - 1) // self.batch_size
        return tot

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        batches = []
        for lo, hi in zip(self.offsets[:-1], self.offsets[1:]):
            idx = np.arange(int(lo), int(hi))
            if self.shuffle:
                rng.shuffle(idx)
            for k in range(0, len(idx), self.batch_size):
                batches.append(idx[k:k + self.batch_size].tolist())
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)


def get_region_dataloader(args, purpose, regions=None):
    """多区域 DataLoader：``(x, y, region_idx)``，batch 内同区域。"""
    ds = MultiRegionDataset(args, purpose, regions=regions)
    sampler = RegionBatchSampler(ds, batch_size=int(getattr(args, 'batch_size', 8)),
                                 shuffle=(purpose == 'train'), seed=0)
    return DataLoader(ds, batch_sampler=sampler, pin_memory=True,
                      num_workers=int(getattr(args, 'num_workers', 4)))


def get_region_dataloaders(args, purpose, regions=None):
    """每个区域单独一个 DataLoader（区域均衡训练用：每步各区域各取一个 batch）。"""
    ds = MultiRegionDataset(args, purpose, regions=regions, verbose=False)
    out = {}
    for i, r in enumerate(ds.regions):
        lo, hi = int(ds.offsets[i]), int(ds.offsets[i + 1])
        idx = list(range(lo, hi))
        bs = int(getattr(args, 'batch_size', 8))
        sampler = RegionBatchSampler(ds, batch_size=bs, shuffle=(purpose == 'train'), seed=i)
        sampler.offsets = np.asarray([lo, hi], dtype=np.int64)   # 只在该区域范围内采
        sampler.region_index = i
        out[r] = (DataLoader(ds, batch_sampler=sampler, pin_memory=True,
                             num_workers=int(getattr(args, 'num_workers', 4))), idx)
    return ds, {r: v[0] for r, v in out.items()}
