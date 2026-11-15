"""区域并行 DDP 训练（4 卡）：每个 rank 负责一个区域，DDP 负责梯度平均。

为什么这样切：
  本工程的多区域模型是"共享主干 + 每区域自己的 adapter/head"，训练的原始目标是
  L = (L_c + L_a + L_n + L_z) / 4（region_balanced 梯度累积）。把 4 个区域分给 4 个 rank、
  让 DDP 对梯度做 all-reduce 平均，得到的更新与单卡"四区梯度累积后 /4 一次 step"**数学等价**，
  但每个 step 的 4 个 batch 是并行算的 → 单卡 ~9 min/epoch 降到 ~2.5 min/epoch。

  * 每个 rank 只前向自己的区域 → 自己区域之外的 head/adapter 在该 rank 上属于"未使用参数"，
    所以必须 find_unused_parameters=True；
  * DDP 的 all-reduce 会把这些"其它 rank 没有梯度"的参数也平均（其余 rank 贡献 0），
    因此各 rank 的副本始终保持一致 → rank0 的 state_dict 就是合并好的完整模型，存盘无需额外合并；
  * 单卡调试（world_size=1）走与 train.py 完全一致的"四区累积"路径，便于回归对比。

用法：
    torchrun --nproc_per_node=4 --standalone train_ddp.py \
        --yaml_config ./config/multiregion_l2_ddp.yaml --config multiregion
    单卡冒烟：torchrun --nproc_per_node=1 --standalone train_ddp.py ... --debug_steps 3
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.YParams import YParams
from utils.compat import get_cosine_schedule_with_warmup
from utils.darcy_loss import LpLoss_weighted
from utils.data_loader import get_region_dataloader
from train import Config, seed_everything
from networks.model import SSEPREModel


def ddp_env():
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    return world, rank, local


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--yaml_config', default='./config/multiregion_l2_ddp.yaml', type=str)
    p.add_argument('--config', default='multiregion', type=str)
    p.add_argument('--debug_steps', default=0, type=int, help='>0 时每 epoch 只跑这么多步（冒烟用）')
    p.add_argument('--resume', default='', type=str, help='从某个 ckpt 继续（可选）')
    return p.parse_args()


class RegionParallelTrainer:
    def __init__(self, params, world, rank, local_rank, debug_steps=0):
        self.params = params
        self.world, self.rank, self.local_rank = world, rank, local_rank
        self.debug_steps = int(debug_steps or 0)
        # SMOKE_SINGLE_GPU=1：多 rank 挤在同一张卡上（仅用于单卡机器上验证 DDP 链路）
        self.smoke_single_gpu = os.environ.get('SMOKE_SINGLE_GPU', '') == '1'
        self.dev_id = 0 if self.smoke_single_gpu else local_rank
        self.device = torch.device('cuda:%d' % self.dev_id)
        torch.cuda.set_device(self.device)
        self.is_main = (rank == 0)

        self.config = Config(params)
        self.regions = list(self.config.region_layouts.keys())
        if world > 1 and world != len(self.regions):
            raise SystemExit('[ddp] world_size=%d 与区域数 %d 不一致；本脚本要求"一个 rank 一个区域"'
                             % (world, len(self.regions)))
        # 该 rank 负责的区域：world==1 时负责全部（等价 train.py 的四区累积）
        self.my_regions = self.regions if world == 1 else [self.regions[rank]]

        self.window = int(getattr(params, 'time_chans', 2))
        self.block_len = int(getattr(params, 'block_len', 1))
        self.n_blocks = int(getattr(params, 'n_blocks', 1)) if params.two_step_training else 1
        self.data_chans = int(params.data_chans)
        self.predict_delta = bool(getattr(self.config, 'predict_delta', False))
        self.delta_weight = float(getattr(self.config, 'delta_weight', 1.0))
        self.loss_mode = str(getattr(params, 'loss_mode', 'l2')).lower()

        # 数据与损失
        self.train_loaders = {r: get_region_dataloader(params, 'train', regions=[r]) for r in self.my_regions}
        self.eval_loaders = {r: get_region_dataloader(params, 'eval', regions=[r]) for r in self.my_regions}
        self.steps_per_epoch = max(len(l) for l in self.train_loaders.values())
        if self.debug_steps:
            self.steps_per_epoch = min(self.steps_per_epoch, self.debug_steps)
        self.loss_objs = {}
        for r, lay in self.config.region_layouts.items():
            self.loss_objs[r] = LpLoss_weighted(
                lay['llh_path'], mask_path=getattr(params, 'mask_path', None),
                mask_key=lay.get('mask_key', r), mask_boost=getattr(params, 'mask_boost', 1.0),
                loss_mode=self.loss_mode)

        # 模型 / DDP
        self.model = SSEPREModel(self.config).to(self.device)
        self.ddp = None
        if world > 1:
            self.ddp = DDP(self.model, device_ids=[self.dev_id], output_device=self.dev_id,
                           find_unused_parameters=True, broadcast_buffers=False)
        self.net = self.ddp if self.ddp is not None else self.model

        # 阶段计划
        self.stage1 = int(getattr(self.config, 'stage1_epochs', 0) or 0)
        self.stage2 = int(getattr(self.config, 'stage2_epochs', 0) or 0)
        self.total_epochs = (self.stage1 + self.stage2) if (self.stage1 and self.stage2) \
            else int(params.max_epochs)
        self.stage = 1
        self.patience = int(params.early_stopping_patience)
        self._build_optim(self.model.parameters(), float(params.lr), self.total_epochs,
                          tag='stage1')

        self.writer = SummaryWriter(params.writer) if self.is_main else None
        self.best_global = 1e9
        self.best_region = {r: 1e9 for r in self.regions}
        self.counter_region = {r: 0 for r in self.regions}
        self.early_stop_counter = 0
        if self.is_main:
            os.makedirs(os.path.dirname(str(params.best_checkpoint_path)), exist_ok=True)
            print('[ddp] world=%d regions(rank->region)=%s | loss=%s | my_regions=%s | %d 步/epoch'
                  % (world, {i: r for i, r in enumerate(self.regions)}, self.loss_mode,
                     self.my_regions, self.steps_per_epoch))

    # --------------------------------------------------------------- optimizer
    def _build_optim(self, params_iter, lr, epochs, tag=''):
        trainable = [p for p in params_iter if p.requires_grad]
        if not trainable:
            raise RuntimeError('没有可训练参数')
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01, eps=1e-6)
        steps = max(1, epochs * self.steps_per_epoch)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(steps * 0.05), num_training_steps=steps)
        if self.is_main:
            print('[optim%s] 可训练张量 %d 个 / %.2fM 参数 | lr=%g | %d 步'
                  % (tag, len(trainable), sum(p.numel() for p in trainable) / 1e6, lr, steps))

    # ------------------------------------------------------------------ forward
    def forward_blocks(self, x, y=None, region=None):
        B, W, C, L = x.shape
        cur = x
        blocks, data_loss, aux_total = [], None, None
        for b in range(self.n_blocks):
            base = cur[:, -1]
            out, aux = self.net(cur, region=region)
            block = out.reshape(B, self.block_len, C, L)
            blocks.append(block)
            aux_total = aux if aux_total is None else aux_total + aux
            if y is not None:
                start = b * self.block_len
                target = y[:, start:start + self.block_len]
                if target.shape[1] == self.block_len:
                    lo = self.loss_objs[region]
                    loss_b = lo(out, target.reshape(B, self.block_len * C, L))
                    if self.predict_delta and self.delta_weight > 0 and \
                            str(getattr(self.params, 'target_mode', 'next_day')) == 'next_day':
                        d_pred = block - base.unsqueeze(1)
                        d_true = target - base.unsqueeze(1)
                        loss_d = lo.rel(d_pred.reshape(B, self.block_len * C, L),
                                        d_true.reshape(B, self.block_len * C, L),
                                        norm=target.reshape(B, self.block_len * C, L))
                        loss_b = loss_b + self.delta_weight * loss_d
                    data_loss = loss_b if data_loss is None else data_loss + loss_b
            cur = torch.cat([cur[:, self.block_len:], block], dim=1)
        if aux_total is None:
            aux_total = torch.zeros((), device=x.device)
        return blocks, data_loss, aux_total

    # -------------------------------------------------------------------- epoch
    def train_epoch(self, epoch):
        self.model.train()
        n_reg = len(self.my_regions)
        iters = {r: iter(self.train_loaders[r]) for r in self.my_regions}
        running = {r: 0.0 for r in self.my_regions}
        total, nb = 0.0, 0
        bar = tqdm(range(self.steps_per_epoch), disable=not self.is_main,
                   desc='train ep%d' % (epoch + 1))
        for _ in bar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for r in self.my_regions:
                try:
                    data = next(iters[r])
                except StopIteration:
                    iters[r] = iter(self.train_loaders[r])
                    data = next(iters[r])
                x, y = (data[0], data[1]) if len(data) >= 3 else (data[0], data[1])
                x = x.to(self.device, non_blocking=True); y = y.to(self.device, non_blocking=True)
                _, data_loss, aux_loss = self.forward_blocks(x, y, region=r)
                loss_r = data_loss + aux_loss * self.config.aux_loss_coef
                # world>1：DDP 会自动对梯度求平均（等价 /n_reg）；world==1：这里手动 /n_reg
                (loss_r if n_reg == 1 else loss_r / n_reg).backward()
                running[r] += float(loss_r.detach())
                step_loss += float(loss_r.detach())
            self.optimizer.step()
            self.scheduler.step()
            total += step_loss / n_reg
            nb += 1
        logs = {'train_loss': total / max(1, nb)}
        for r in self.my_regions:
            logs['train_loss_' + r] = running[r] / max(1, nb)
        return logs

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()
        local = {}
        for r in self.my_regions:
            tot, sk, ps, n = 0.0, 0.0, 0.0, 0
            lo = self.loss_objs[r]
            for data in self.eval_loaders[r]:
                x, y = data[0].to(self.device), data[1].to(self.device)
                blocks, data_loss, aux_loss = self.forward_blocks(x, y, region=r)
                B = x.shape[0]
                tot += float((data_loss + aux_loss * self.config.aux_loss_coef)) * B
                for b, blk in enumerate(blocks):
                    tgt = y[:, b * self.block_len:(b + 1) * self.block_len]
                    if tgt.shape[1] != self.block_len:
                        continue
                    t_ = tgt.reshape(B, -1, tgt.shape[-1])
                    sk += float(lo.rel(blk.reshape(B, -1, blk.shape[-1]), t_)) * B
                    base = x[:, -1].unsqueeze(1).expand(-1, self.block_len, -1, -1)
                    ps += float(lo.rel(base.reshape(B, -1, base.shape[-1]), t_)) * B
                n += B
            nb = max(1, len(blocks)) * max(1, n)
            local[r] = (tot / max(1, n), sk / nb, ps / nb)

        # 汇总：把 (eval_loss, skill, persist) 按 rank 收集
        world = self.world
        buf = torch.zeros(3 * world, device=self.device, dtype=torch.float64)
        if world > 1:
            mean_local = np.mean([v[0] for v in local.values()])
            # 每 rank 只报自己区域的三个量；用 region 顺序重排
            r0 = self.my_regions[0]
            idx = self.regions.index(r0)
            buf[3 * idx:3 * idx + 3] = torch.tensor(local[r0], dtype=torch.float64, device=self.device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            vals = buf.tolist()
            for i, r in enumerate(self.regions):
                local[r] = tuple(vals[3 * i:3 * i + 3])
        logs = {'eval_loss': float(np.mean([v[0] for v in local.values()]))}
        for r, (loss_r, sk, ps) in local.items():
            logs['eval_loss_' + r] = loss_r
            logs['eval_skill_' + r] = sk
            logs['eval_persist_' + r] = ps
            logs['eval_gain_' + r] = 100.0 * (1 - sk / max(ps, 1e-9))
        gains = [100.0 * (1 - local[r][1] / max(local[r][2], 1e-9)) for r in local]
        logs['eval_gain_mean'] = float(np.mean(gains))
        return logs, local

    # --------------------------------------------------------------------- save
    def save(self, path):
        if not self.is_main:
            return
        d = os.path.dirname(str(path))
        if d:
            os.makedirs(d, exist_ok=True)
        meta = {'predict_delta': self.predict_delta,
                'patch_size': int(self.config.patch_size),
                'time_chans': int(self.config.time_chans),
                'block_len': int(self.block_len), 'n_blocks': int(self.n_blocks),
                'pos_embed_mode': str(self.config.pos_embed_mode),
                'regions': list(self.regions), 'ddp_world': self.world}
        torch.save({'model_state': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'meta': meta}, path)

    # ------------------------------------------------------------------ stages
    def switch_to_stage2(self):
        if self.is_main and self.best_global < 1e5:
            self.save(os.path.join(os.path.dirname(str(self.params.best_checkpoint_path)),
                                   'global_best.ckpt'))
            print('[stage2] 阶段一 best=%.4f 已另存 global_best.ckpt' % self.best_global)
        self.best_global = 1e9
        self.counter_region = {r: 0 for r in self.regions}
        if int(getattr(self.config, 'stage2_patience', 0) or 0) > 0:
            self.patience = int(self.config.stage2_patience)
        keep_adapters = 'adapter' in str(self.config.stage2_train)
        # 冻结共享主干：region 模块之外全部 requires_grad=False
        prefixes = ['region_heads.']
        if keep_adapters:
            prefixes.append('region_adapters.')
        if not bool(self.config.freeze_region_embed):
            prefixes.append('region_embed.')
        for name, p in self.model.named_parameters():
            p.requires_grad_(any(name.startswith(q) for q in prefixes))
        if self.ddp is not None:
            self.ddp = DDP(self.model, device_ids=[self.dev_id], output_device=self.dev_id,
                           find_unused_parameters=True, broadcast_buffers=False)
            self.net = self.ddp
        lr2 = self.config.stage2_lr if self.config.stage2_lr > 0 else float(self.params.lr)
        remaining = max(1, self.stage2 or (self.total_epochs - self.stage1))
        self._build_optim(self.model.parameters(), float(lr2), remaining, tag='stage2')
        self.stage = 2
        if self.is_main:
            print('[stage2] 冻结共享主干，只训 %s | patience=%d' % (prefixes, self.patience))

    # -------------------------------------------------------------------- train
    def train(self):
        t0 = time.time()
        for epoch in range(self.total_epochs):
            if self.stage1 and self.stage == 1 and epoch >= self.stage1:
                self.switch_to_stage2()
            tl = self.train_epoch(epoch)
            el, local = self.eval_epoch()
            self.save(self.params.checkpoint_path)
            if self.is_main:
                print('ep%02d | train %.4f | eval %s | gain_mean %+.1f%% | %.1f min'
                      % (epoch + 1, tl['train_loss'],
                         ' '.join('%s=%.4f' % (r[:4], local[r][0]) for r in self.regions),
                         el['eval_gain_mean'], (time.time() - t0) / 60))
            if el['eval_loss'] < self.best_global:
                self.best_global = el['eval_loss']
                self.save(self.params.best_checkpoint_path)
            # 阶段二：每区独立选 best / 早停
            stop = False
            if self.stage == 2 and bool(self.config.stage2_per_region_best):
                for r in self.regions:
                    cur = el.get('eval_loss_' + r)
                    if cur is None:
                        continue
                    if cur < self.best_region[r]:
                        self.best_region[r] = cur
                        self.counter_region[r] = 0
                        self.save(os.path.join(os.path.dirname(str(self.params.best_checkpoint_path)),
                                               'region_%s_best.ckpt' % r))
                    else:
                        self.counter_region[r] += 1
                if self.is_main:
                    print('      per-region best | ' + ' '.join(
                        '%s=%.4f(ni %d)' % (r[:4], self.best_region[r], self.counter_region[r])
                        for r in self.regions))
                stop = all(c >= self.patience for c in self.counter_region.values())
            else:
                if el['eval_loss'] < self.best_global - 1e-6:
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    stop = self.early_stop_counter >= self.patience
            # 所有 rank 必须一起停（否则 DDP 挂死）
            flag = torch.tensor([1.0 if stop else 0.0], device=self.device)
            if self.world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item() > 0:
                if self.is_main:
                    print('早停于 epoch %d' % (epoch + 1))
                break
            if self.is_main:
                self.writer.add_scalar('train_loss', tl['train_loss'], epoch)
                self.writer.add_scalar('eval_loss', el['eval_loss'], epoch)
                self.writer.add_scalar('eval_gain_mean', el['eval_gain_mean'], epoch)
                for k in ('eval_gain_cascadia', 'eval_gain_alaska', 'eval_gain_nankai',
                          'eval_gain_newzealand', 'eval_loss_cascadia', 'eval_loss_alaska',
                          'eval_loss_nankai', 'eval_loss_newzealand'):
                    if k in el:
                        self.writer.add_scalar(k, el[k], epoch)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        if self.is_main:
            print('训练结束，用时 %.1f min' % ((time.time() - t0) / 60))


def main():
    args = parse_args()
    world, rank, local = ddp_env()
    _smoke = os.environ.get('SMOKE_SINGLE_GPU', '') == '1'
    if world > 1:
        # 正式任务用 nccl；单卡机器上多 rank 冒烟时 NCCL 无法在"同一张卡"上建通信域，
        # 可用 DDP_BACKEND=gloo 退化成 gloo（只影响冒烟，不影响 4 卡真实任务）。
        dist.init_process_group(backend=os.environ.get('DDP_BACKEND', 'nccl'))
        torch.cuda.set_device(0 if _smoke else local)
    seed_everything(111)          # 所有 rank 同种子 → 初始权重逐位一致
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    tr = RegionParallelTrainer(params, world, rank, local, debug_steps=args.debug_steps)
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        tr.model.load_state_dict(ck['model_state'])
        if rank == 0:
            print('[resume] loaded', args.resume)
    tr.train()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
