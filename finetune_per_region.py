"""按区域逐个微调各区域的 adapter / head，最后合并成一个 ckpt。

用途（多区域预训练 -> 单区域微调）：
    1. 用 config/multiregion.yaml 建**完整**多区域模型（四个区域都在，架构不能缩，否则权重对不上）；
    2. 加载多区域预训练权重（best_model.ckpt / global_best.ckpt / merged_per_region_best.ckpt 都行）；
    3. 对 ft_regions 里每个区域 r 依次：
         - 只放开 ``region_adapters.<r>.*`` 与 ``region_heads.<r>.*``；共享主干、
           region_embed、head1/head2/head_out、**其它区域的头** 全部冻结，不会被动到；
         - 只用 r 自己的 train 划分训练、r 自己的 eval 划分选优与早停；
         - 保存该区域模块的最优快照（<out>/<tag>_<region>.pt）；
    4. 把各区最优模块合并回预训练主干，落成**一个** ckpt（结构与 train.py 产出完全一致，
       ipynb/predict_cascadia.ipynb 里的 Load_model 不用改就能加载）。

用法：
    python finetune_per_region.py --ft_config l2 --tag cascadia_l2 --ft_regions cascadia
    python finetune_per_region.py --ft_config l2 --tag all_l2        # 四区依次微调后合并

产物：
    <out_dir>/<tag>.ckpt             合并后的单一模型（可直接给推理 notebook 用）
    <out_dir>/<tag>_<region>.pt      各区域最优模块快照（可单独回滚/重组）
    <out_dir>/<tag>_summary.json     每区指标（训练口径 / 另一种口径 / persistence 对照）
    <tb_dir>/<tag>                   tensorboard（每区一组曲线）
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.YParams import YParams
from utils.compat import get_cosine_schedule_with_warmup
from utils.darcy_loss import LpLoss_weighted
from utils.data_loader import get_region_dataloader
from train import Config, seed_everything
from networks.model import SSEPREModel


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_yaml', default='./config/multiregion.yaml', type=str)
    p.add_argument('--base_config', default='multiregion', type=str)
    p.add_argument('--ft_yaml', default='./config/finetune_per_region.yaml', type=str)
    p.add_argument('--ft_config', default='l2', type=str)
    p.add_argument('--tag', default='', type=str, help='产物前缀，默认用 ft_config')
    p.add_argument('--ft_regions', default='', type=str,
                   help='逗号分隔，覆盖 yaml 里的 ft_regions；例如 cascadia 或 cascadia,alaska')
    p.add_argument('--pretrained_ckpt_path', default='', type=str)
    p.add_argument('--loss_mode', default='', type=str, choices=['', 'l1', 'l2'])
    p.add_argument('--epochs', default=None, type=int)
    p.add_argument('--lr', default=None, type=float)
    p.add_argument('--patience', default=None, type=int)
    p.add_argument('--keep_adapters', default=None, type=str, choices=['true', 'false'])
    p.add_argument('--out_dir', default='', type=str, help='覆盖 yaml 的 out_dir')
    p.add_argument('--tb_dir', default='', type=str, help='覆盖 yaml 的 tb_dir')
    p.add_argument('--seed', default=111, type=int)
    return p.parse_args()


def build_params(args):
    """多区域预训练配置 + 微调覆盖项。

    注意：**绝不能改 params.regions**。架构（每区的 token 布局、region_heads 的行数）
    必须和预训练 ckpt 一致，否则 state_dict 对不上。要微调哪些区域由 ft_regions 单独控制。
    """
    params = YParams(args.base_yaml, args.base_config)
    overrides = YParams(args.ft_yaml, args.ft_config).params
    for k, v in overrides.items():
        setattr(params, k, v)
    # 本脚本自带训练循环，关掉 train.py 的两阶段调度（Trainer 不参与）
    params.stage1_epochs = 0
    params.stage2_epochs = 0
    # 命令行覆盖
    if args.pretrained_ckpt_path:
        params.pretrained_ckpt_path = args.pretrained_ckpt_path
    if args.loss_mode:
        params.loss_mode = args.loss_mode
    if args.epochs is not None:
        params.max_epochs = args.epochs
    if args.lr is not None:
        params.lr = args.lr
    if args.patience is not None:
        params.early_stopping_patience = args.patience
    if args.keep_adapters is not None:
        params.keep_adapters = (args.keep_adapters == 'true')
    if args.out_dir:
        params.out_dir = args.out_dir
        if not args.tb_dir:          # 默认把 tensorboard 收进同一目录
            params.tb_dir = os.path.join(args.out_dir, 'tensorboard')
    if args.tb_dir:
        params.tb_dir = args.tb_dir
    if args.ft_regions:
        params.ft_regions = [x.strip() for x in args.ft_regions.split(',') if x.strip()]
    return params


def unpack_batch(data):
    """dataloader: (x, y, region_idx) 多区域 / (x, y) 单区域。region 由调用方显式指定。"""
    if len(data) == 3:
        x, y, _rid = data
        return x, y
    x, y = data
    return x, y


# ----------------------------------------------------------------------- trainer
class PerRegionFinetuner:
    def __init__(self, params):
        self.params = params
        self.device = torch.device(params.device)
        self.config = Config(params)
        self.loss_mode = str(getattr(params, 'loss_mode', 'l2')).lower()
        assert self.loss_mode in ('l1', 'l2')
        self.alt_mode = 'l2' if self.loss_mode == 'l1' else 'l1'
        self.keep_adapters = bool(getattr(params, 'keep_adapters', True))

        self.window = int(getattr(params, 'time_chans', 2))
        self.block_len = int(getattr(params, 'block_len', 1))
        self.n_blocks = int(getattr(params, 'n_blocks', 1)) if params.two_step_training else 1
        self.data_chans = int(params.data_chans)
        self.predict_delta = bool(getattr(self.config, 'predict_delta', False))
        self.delta_weight = float(getattr(self.config, 'delta_weight', 1.0))
        self.delta_on = self.predict_delta and self.delta_weight > 0 and \
            str(getattr(params, 'target_mode', 'next_day')) == 'next_day'
        self.next_day = str(getattr(params, 'target_mode', 'next_day')) == 'next_day'

        # 架构层面的区域列表（= ckpt 里的区域顺序），不可裁剪
        self.arch_regions = list(self.config.region_layouts.keys())
        self.model = SSEPREModel(self.config).to(self.device)

        self.loss_objs = {}
        self.loss_objs_alt = {}
        for r, lay in self.config.region_layouts.items():
            kw = dict(mask_path=getattr(params, 'mask_path', None),
                      mask_key=lay.get('mask_key', r),
                      mask_boost=getattr(params, 'mask_boost', 1.0))
            self.loss_objs[r] = LpLoss_weighted(lay['llh_path'], loss_mode=self.loss_mode, **kw)
            self.loss_objs_alt[r] = LpLoss_weighted(lay['llh_path'], loss_mode=self.alt_mode, **kw)

        self.out_dir = str(getattr(params, 'out_dir', './Results/multiregion/finetune_per_region'))
        self.tb_dir = str(getattr(params, 'tb_dir', os.path.join(self.out_dir, 'tensorboard')))
        os.makedirs(self.out_dir, exist_ok=True)

        self.base_state = None
        self.load_pretrained()

    # ------------------------------------------------------------------ ckpt
    def load_pretrained(self):
        path = str(getattr(self.params, 'pretrained_ckpt_path', '') or '')
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f'pretrained_ckpt_path 不存在: {path!r}')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        state = ck.get('model_state', ck)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f'与预训练权重不匹配：missing={len(missing)} unexpected={len(unexpected)}。\n'
                f'missing[:5]={missing[:5]}\nunexpected[:5]={unexpected[:5]}\n'
                '本脚本只能微调"同架构"的多区域权重；请确认 base_yaml 与产出该 ckpt 的配置一致。')
        self.base_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        meta = ck.get('meta') or {}
        print(f'[load] {path}')
        print(f'[load] meta={meta}')
        print(f'[load] loss_mode={self.loss_mode} (对照口径 {self.alt_mode}) | '
              f'keep_adapters={self.keep_adapters} | blocks={self.n_blocks} | delta={self.predict_delta}')

    # ------------------------------------------------------------- forward/loss
    def forward_blocks(self, x, y=None, region=None):
        """与 train.py::Trainer.forward_blocks 逐行等价（保证微调与预训练口径一致）。"""
        B, W, C, L = x.shape
        cur = x
        blocks, data_loss, aux_total = [], None, None
        for b in range(self.n_blocks):
            base = cur[:, -1]
            out, aux = self.model(cur, region=region)
            block = out.reshape(B, self.block_len, C, L)
            blocks.append(block)
            aux_total = aux if aux_total is None else aux_total + aux
            if y is not None:
                start = b * self.block_len
                target = y[:, start:start + self.block_len]
                if target.shape[1] == self.block_len:
                    lo = self.loss_objs[region]
                    loss_b = lo(out, target.reshape(B, self.block_len * C, L))
                    if self.delta_on:
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

    def score_with_persistence(self, x, blocks, y, region):
        """模型精度 + persistence 基线（同一口径），返回 (rel, rel_alt, persist, persist_alt)。"""
        B = y.shape[0]
        res = {}
        for name, objs in (('', self.loss_objs), ('_alt', self.loss_objs_alt)):
            lo = objs[region]
            m_sum, p_sum, n_blk = 0.0, 0.0, 0
            for b, blk in enumerate(blocks):
                start = b * self.block_len
                tgt = y[:, start:start + self.block_len]
                if tgt.shape[1] != self.block_len:
                    continue
                t_ = tgt.reshape(B, -1, tgt.shape[-1])
                p_ = blk.reshape(B, -1, blk.shape[-1])
                base = x[:, -1].unsqueeze(1).expand(-1, self.block_len, -1, -1)
                base_ = base.reshape(B, -1, base.shape[-1])
                m_sum += float(lo.rel(p_, t_)) * B
                p_sum += float(lo.rel(base_, t_)) * B
                n_blk += 1
            den = max(1, n_blk * B)
            res['rel' + name] = m_sum / den
            res['persist' + name] = p_sum / den
        return res

    # ------------------------------------------------------------------ freeze
    def set_trainable(self, region):
        prefixes = ['region_heads.%s.' % region]
        if self.keep_adapters:
            prefixes.append('region_adapters.%s.' % region)
        for name, p in self.model.named_parameters():
            p.requires_grad_(any(name.startswith(q) for q in prefixes))
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f'区域 {region} 没有可训练参数：请检查 region_head / region_adapter 是否为 True')
        # 断言：放开的名字必须都属于本区域
        bad = [n for n, _ in trainable if not any(n.startswith(q) for q in prefixes)]
        assert not bad, bad
        n_par = sum(p.numel() for _, p in trainable)
        print(f'[freeze] region={region} | 可训练张量 {len(trainable)} 个 / {n_par/1e3:.1f}K 参数 | '
              f'其余 {sum(p.numel() for n,p in self.model.named_parameters() if not p.requires_grad)/1e6:.2f}M 全部冻结')
        return [p for _, p in trainable]

    def region_module_state(self, region):
        """该区域的 adapter/head 参数（用于选优快照与合并）。"""
        keys = ['region_heads.%s.' % region]
        if self.keep_adapters:
            keys.append('region_adapters.%s.' % region)
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()
                if any(k.startswith(q) for q in keys)}

    def load_region_module_state(self, region, module_state):
        cur = self.model.state_dict()
        for k, v in module_state.items():
            assert k in cur and cur[k].shape == v.shape, k
            cur[k] = v.to(cur[k].device)
        self.model.load_state_dict(cur)

    # -------------------------------------------------------------------- eval
    @torch.no_grad()
    def eval_region(self, region, loader):
        self.model.eval()
        tot = {'rel': 0.0, 'rel_alt': 0.0, 'persist': 0.0, 'persist_alt': 0.0}
        n = 0
        for data in loader:
            x, y = unpack_batch(data)
            x = x.to(self.device); y = y.to(self.device)
            blocks, _, _ = self.forward_blocks(x, y, region=region)
            res = self.score_with_persistence(x, blocks, y, region)
            B = x.shape[0]
            for k in tot:
                tot[k] += res[k] * B
            n += B
        out = {k: v / max(1, n) for k, v in tot.items()}
        out['gain'] = 100.0 * (1 - out['rel'] / max(out['persist'], 1e-9))
        out['gain_alt'] = 100.0 * (1 - out['rel_alt'] / max(out['persist_alt'], 1e-9))
        return out

    # ------------------------------------------------------------------- train
    def train_region(self, region, writer=None):
        params = self.params
        epochs = int(getattr(params, 'max_epochs', 40))
        lr = float(getattr(params, 'lr', 5e-4))
        patience = int(getattr(params, 'early_stopping_patience', 8))
        loader = get_region_dataloader(params, 'train', regions=[region])
        eloader = get_region_dataloader(params, 'eval', regions=[region])
        trainable = self.set_trainable(region)

        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01, eps=1e-6)
        steps = max(1, epochs * len(loader))
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(steps * 0.05),
                                                num_training_steps=steps)
        print(f'[ft:{region}] {len(loader)} 步/epoch × {epochs} epoch | lr={lr:g} | patience={patience}')

        best = None
        best_metric = 1e9
        counter = 0
        history = []
        for ep in range(epochs):
            self.model.train()
            running, nb = 0.0, 0
            bar = tqdm(loader, desc=f'ft[{region}] epoch[{ep+1}/{epochs}]', leave=False)
            for data in bar:
                x, y = unpack_batch(data)
                x = x.to(self.device); y = y.to(self.device)
                opt.zero_grad(set_to_none=True)
                _, data_loss, aux_loss = self.forward_blocks(x, y, region=region)
                loss = data_loss + aux_loss * self.config.aux_loss_coef
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step()
                running += float(loss.detach()); nb += 1
                bar.set_postfix(loss=f'{running/max(1,nb):.4f}')
            ev = self.eval_region(region, eloader)
            ev['epoch'] = ep + 1
            ev['train_loss'] = running / max(1, nb)
            history.append(ev)
            print('[ft:%s] epoch %2d | train %.4f | eval %s=%.4f (alt %s=%.4f) | '
                  'persist %.4f | gain %+.1f%% | lr %.2e'
                  % (region, ep + 1, ev['train_loss'], self.loss_mode, ev['rel'],
                     self.alt_mode, ev['rel_alt'], ev['persist'], ev['gain'],
                     opt.param_groups[0]['lr']))
            if writer is not None:
                pre = f'ft/{region}'
                for k in ('train_loss', 'rel', 'rel_alt', 'persist', 'persist_alt', 'gain', 'gain_alt'):
                    writer.add_scalar(f'{pre}/{k}', ev[k], ep)
            # 选优：用训练口径的纯精度项（与预训练/eval_one_epoch 的 skill 一致）
            if ev['rel'] < best_metric - 1e-5:
                best_metric = ev['rel']
                best = {'epoch': ep + 1, 'eval': ev, 'module_state': self.region_module_state(region)}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'[ft:{region}] 早停于 epoch {ep+1}（{patience} 轮无改善），best epoch={best["epoch"]}')
                    break
        if best is None:
            raise RuntimeError(f'区域 {region} 没有产生任何最优快照')
        self.load_region_module_state(region, best['module_state'])
        return best, history

    # ------------------------------------------------------------------- merge
    def merge_and_save(self, snapshots, tag, extra_meta=None):
        state = {k: v.detach().clone() for k, v in self.base_state.items()}
        allowed, touched = [], []
        for region, snap in snapshots.items():
            allowed += ['region_heads.%s.' % region]
            if self.keep_adapters:
                allowed.append('region_adapters.%s.' % region)
            for k, v in snap.items():
                assert k in state and state[k].shape == v.shape, k
                state[k] = v
        for k in state:
            if not torch.equal(state[k], self.base_state[k]):
                touched.append(k)
        bad = [k for k in touched if not any(k.startswith(a) for a in allowed)]
        if bad:
            raise RuntimeError(f'合并时出现了不该被改动的张量: {bad[:5]}')
        print(f'[merge] 改动的张量 {len(touched)} 个，全部属于 {sorted(snapshots)} 的区域模块'
              f'（其余 {len(state)-len(touched)} 个与预训练逐位相同）')
        meta = {'predict_delta': self.predict_delta,
                'patch_size': int(self.config.patch_size),
                'time_chans': int(self.config.time_chans),
                'block_len': int(self.block_len),
                'n_blocks': int(self.n_blocks),
                'pos_embed_mode': str(self.config.pos_embed_mode),
                'regions': list(self.arch_regions),
                'finetune_per_region': dict(extra_meta or {})}
        out_path = os.path.join(self.out_dir, f'{tag}.ckpt')
        torch.save({'model_state': state, 'optimizer_state_dict': {}, 'meta': meta}, out_path)
        # 严格自检：合并后的 ckpt 能被同架构模型严格加载
        chk = SSEPREModel(self.config)
        chk.load_state_dict(torch.load(out_path, map_location='cpu', weights_only=False)['model_state'])
        print(f'[merge] 已写出（严格加载自检通过）: {out_path}')
        return out_path


def main():
    args = parse_args()
    seed_everything(args.seed)
    params = build_params(args)
    tag = args.tag or args.ft_config
    ft_regions = list(getattr(params, 'ft_regions', None) or [])
    print(f'[cfg] base={args.base_yaml}:{args.base_config} | ft={args.ft_yaml}:{args.ft_config} | tag={tag}')
    print(f'[cfg] pretrained={params.pretrained_ckpt_path}')
    print(f'[cfg] ft_regions={ft_regions} | loss={params.loss_mode} | lr={params.lr} | '
          f'epochs={params.max_epochs} | patience={params.early_stopping_patience} | '
          f'keep_adapters={getattr(params, "keep_adapters", True)}')

    ft = PerRegionFinetuner(params)
    unknown = [r for r in ft_regions if r not in ft.arch_regions]
    if unknown:
        raise ValueError(f'ft_regions 里有未知区域 {unknown}；可选 {ft.arch_regions}')

    writer = torch.utils.tensorboard.SummaryWriter(os.path.join(ft.tb_dir, tag))
    snapshots, summary = {}, {}
    t0 = time.time()
    for region in ft_regions:
        print('=' * 78)
        print(f'[ft] === {region} ===')
        best, history = ft.train_region(region, writer=writer)
        snapshots[region] = best['module_state']
        torch.save({'module_state': best['module_state'], 'meta': {
            'region': region, 'best_epoch': best['epoch'], 'eval': best['eval'],
            'pretrained_ckpt_path': str(params.pretrained_ckpt_path),
            'loss_mode': ft.loss_mode, 'keep_adapters': ft.keep_adapters}},
            os.path.join(ft.out_dir, f'{tag}_{region}.pt'))
        summary[region] = {'best_epoch': best['epoch'], 'best_eval': best['eval'],
                           'history': history}
        print(f'[ft] {region} 完成，best epoch={best["epoch"]}，用时 {(time.time()-t0)/60:.1f} min')

    out_path = ft.merge_and_save(snapshots, tag, extra_meta={
        'pretrained_ckpt_path': str(params.pretrained_ckpt_path),
        'loss_mode': ft.loss_mode, 'keep_adapters': ft.keep_adapters,
        'regions_finetuned': list(snapshots), 'lr': float(params.lr),
        'epochs': int(params.max_epochs), 'patience': int(params.early_stopping_patience)})
    with open(os.path.join(ft.out_dir, f'{tag}_summary.json'), 'w') as f:
        json.dump({'tag': tag, 'ckpt': out_path, 'base_yaml': args.base_yaml,
                   'base_config': args.base_config, 'regions': summary}, f, indent=2,
                  ensure_ascii=False)
    print(f'[done] 全部完成，合并模型: {out_path} | 总用时 {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
