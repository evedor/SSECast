import os
import argparse
import torch
import random
import numpy as np
from utils.YParams import YParams
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.darcy_loss import LpLoss_weighted
from utils.data_loader import get_dataloader, get_region_dataloader
from networks.model import SSEPREModel
from utils.compat import PretrainedConfig, get_cosine_schedule_with_warmup


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config(PretrainedConfig):
    def __init__(self, params,
                 # fea_len=3339,
                 time_chans=None,   # 输入窗口天数：默认取 yaml 里的 time_chans（现为 7 天）
                 data_chans=3,
                 out_chans=3,
                 patch_size=None,  # 默认取 yaml 里的 patch_size（现为 2：一个四边形=2 个三角形）
                 hidden_size=256,
                 num_attention_heads=16,
                 num_key_value_heads=8,
                 flash_attn=True,
                 attention_bias=False,
                 max_seq_len=1113,
                 intermediate_size=1024,
                 mlp_bias=True,
                 dropout=0.05,
                 expert_num=4,
                 topk=2,
                 output_router_logits=True,
                 aux_loss_coef=0.01,
                 **kwargs):
        self.params = params
        # regions 顺序 = 训练/评估顺序；不写 regions 时自动取 region_configs 的全部键
        _rc = getattr(self.params, 'region_configs', None) or {}
        self.regions = list(getattr(self.params, 'regions', None) or list(_rc.keys()))
        self.multiregion = bool(self.regions)
        # 单区域字段：多区域时由各区域布局覆盖（fea_len = 各区域最大），所以这里允许缺省
        self.fea_len = int(getattr(self.params, 'fea_len', 0) or 0)
        # 输入窗长度以 yaml 为准（7 天）；patch_embed 的输入通道 = time_chans * data_chans
        self.time_chans = int(getattr(self.params, 'time_chans', 2) if time_chans is None else time_chans)
        self.data_chans = self.params.data_chans
        # patch_size 也以 yaml 为准（原来硬编码 3，与 params.patch_size 不一致会导致 conv/RoPE 对不上）
        self.patch_size = int(getattr(self.params, 'patch_size', patch_size) if patch_size is None else patch_size)
        self.out_chans = self.params.out_chans
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = (self.fea_len // self.patch_size) if self.fea_len else int(
            getattr(self.params, 'max_seq_len', 1113))   # 多区域后面会被 max(token 数) 覆盖

        # ---- token 条带对齐（方案 A）--------------------------------------------
        # 默认 patch 直接按文件行序每 patch_size 个切一块，会有 ~6% 的 token 横跨断层的
        # "沿倾角条带"接缝（跨 100~400 km）。align_tokens=True 时把每条条带补齐到 patch_size
        # 的整数倍再切，使每个 token 都落在同一块空间里；token 数 1113 -> 1137。
        self.align_tokens = bool(getattr(self.params, 'align_tokens', False))
        self.token_slot_index = None
        self.token_unpad_index = None
        self.token_pad_mask = None
        self.token_n_slots = None
        if self.align_tokens and not self.multiregion:   # 多区域的布局在下面按区域单独建
            from utils.fault_tokens import build_layout
            layout = build_layout(
                self.params.fault_path,
                int(self.params.fea_len),
                patch_size=int(self.params.patch_size),
                jump_km=float(getattr(self.params, 'token_jump_km', 30.0)),
                order=str(getattr(self.params, 'token_order', 'serpentine')),
                pairing=str(getattr(self.params, 'token_pairing', 'phase')),
            )
            self.n_tokens = layout['n_tokens']
            self.token_uv = layout['token_uv']                    # (n_tokens, 2) 几何坐标 (u, v)
            self.token_lonlat = layout['token_lonlat']            # (n_tokens, 2) 平均经纬度 (lon, lat)
            self.token_slot_index = layout['slot_to_element']
            self.token_unpad_index = layout['unpad_index']
            self.token_pad_mask = layout['pad_mask']
            self.token_n_slots = layout['n_slots']
            self.max_seq_len = self.n_tokens
        elif not self.multiregion:
            self.n_tokens = int(self.fea_len) // int(self.params.patch_size)
            self.token_uv = None
            self.token_lonlat = None

        # ---- 多区域：regions 非空时，每个区域一套布局；proj/主干/head/网格 PE 全区域共享 ----
        self.region_layouts = None
        if self.multiregion:
            from utils.region_registry import build_regions
            grid = (int(getattr(self.params, 'pos_grid_lat', 48)),
                    int(getattr(self.params, 'pos_grid_lon', 48)))
            cache_dir = os.path.join(str(getattr(self.params, 'exper_path', './Results')), '_region_cache')
            self.region_layouts = build_regions(
                self.params, self.regions, cache_dir=cache_dir,
                patch_size=int(self.params.patch_size),
                jump_km=float(getattr(self.params, 'token_jump_km', 30.0)),
                order=str(getattr(self.params, 'token_order', 'strip')),
                pairing=str(getattr(self.params, 'token_pairing', 'diagonal')),
                grid=grid,
            )
            self.region_names = list(self.region_layouts.keys())
            self.n_tokens = max(d['n_tokens'] for d in self.region_layouts.values())
            self.fea_len = max(d['fea_len'] for d in self.region_layouts.values())
            self.max_seq_len = int(self.n_tokens)
            self.token_uv = None
            self.token_lonlat = None
            self.token_slot_index = None
            self.token_unpad_index = None
            self.pos_grid_v, self.pos_grid_u = grid
            self.pos_embed_grid_shapes = {r: (d['grid_h'], d['grid_w']) for r, d in self.region_layouts.items()}

        # ---- 位置编码：table(每个 token 一个向量) / grid(二维几何网格+双线性插值) ----
        self.pos_embed_mode = str(getattr(self.params, 'pos_embed_mode', 'table'))
        # grid 的坐标体系：uv(沿倾角弧长/条带号) 或 lonlat(token 平均经纬度)
        self.pos_coord_mode = ('region' if self.regions
                               else str(getattr(self.params, 'pos_coord_mode', 'uv')).lower())
        self.region_embed = bool(getattr(self.params, 'region_embed', False))
        self.region_embed_init = str(getattr(self.params, 'region_embed_init', 'onehot')).lower()
        self.region_embed_scale = float(getattr(self.params, 'region_embed_scale', 0.02))
        # ---- 区域专属 Adapter / Head + 区域均衡梯度累积 + 两阶段训练 ----
        self.region_adapter = bool(getattr(self.params, 'region_adapter', False))
        self.region_adapter_ratio = int(getattr(self.params, 'region_adapter_ratio', 4))
        self.region_head = bool(getattr(self.params, 'region_head', False))
        self.region_balanced = bool(getattr(self.params, 'region_balanced', True))
        self.stage1_epochs = int(getattr(self.params, 'stage1_epochs', 0))
        # 阶段二独立的 epoch 预算与 patience（0 = 沿用 max_epochs / early_stopping_patience）
        self.stage2_epochs = int(getattr(self.params, 'stage2_epochs', 0) or 0)
        self.stage2_patience = int(getattr(self.params, 'stage2_patience', 0) or 0)
        # 阶段二训哪些区域模块：heads+adapters（默认）/ heads（只训各区域 head = 线性探针）
        self.stage2_train = str(getattr(self.params, 'stage2_train', 'heads+adapters')).lower()
        self.stage2_lr = float(getattr(self.params, 'stage2_lr', 0.0) or 0.0)
        # 阶段二是否按区域独立选 best / 早停（各存 region_<r>_best.ckpt）
        self.stage2_per_region_best = bool(getattr(self.params, 'stage2_per_region_best', True))
        self.freeze_region_embed = bool(getattr(self.params, 'freeze_region_embed', True))
        self.pos_grid_v = int(getattr(self.params, 'pos_grid_v', 72))     # uv 模式：沿走向
        self.pos_grid_u = int(getattr(self.params, 'pos_grid_u', 24))     # uv 模式：沿倾角
        self.pos_grid_lat = int(getattr(self.params, 'pos_grid_lat', 48))  # lonlat 模式：行
        self.pos_grid_lon = int(getattr(self.params, 'pos_grid_lon', 24))  # lonlat 模式：列
        # ---- 增量预测：模型输出 Δ，绝对场 = 窗口最后一天 + Δ（见 networks/model.py: forward）----
        self.predict_delta = bool(getattr(self.params, 'predict_delta', False))
        self.delta_weight = float(getattr(self.params, 'delta_weight', 1.0))
        if self.regions:      # 网格尺寸以多区域配置为准（上面 pos_embed 块会被 yaml 的 pos_grid_v/u 覆盖）
            self.pos_grid_v, self.pos_grid_u = int(getattr(self.params, 'pos_grid_lat', 48)), \
                                              int(getattr(self.params, 'pos_grid_lon', 48))
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)


class Trainer():
    """7 天输入 -> 一次预测 7 天 -> 再用预测出的 7 天预测第 8-14 天。

    一个样本的前向过程（``time_chans=2, block_len=1, n_blocks=2``，训练时滚 2 步）：
        step1: x  = data[i   : i+2]                       -> p1 (B,C,L)  目标 data[i+2]（第 3 天）
        step2: x' = data[i+1 : i+3] 中的"第 2 天 + p1"      -> p2 (B,C,L)  目标 data[i+3]（第 4 天）
        即 x' = cat(data[i+1], p1)，窗口右移 1 天，用预测的那一天替换真值。
        loss  = L(p1, y1) + L(p2, y2) (+ router aux loss)
    每次前向只输出 1 天；窗口每步右移 ``block_len`` 天（与 time_chans 无关）。
    ``two_step_training=False`` 只训练 step1。
    """

    def __init__(self, params):
        self.params = params
        self.exper_path = params.exper_path
        # 多区域 yaml 里没有顶层 data_path（路径都在 region_configs 里）→ 取第一个区域做标识
        if getattr(params, 'data_path', None):
            self.exper_data_path = params.data_path
        else:
            _rs = list(getattr(params, 'regions', None) or
                       (getattr(params, 'region_configs', None) or {}).keys())
            from utils.region_registry import region_data_paths
            self.exper_data_path = region_data_paths(params, _rs[0])['data_path'] if _rs else None
        self.device = torch.device(params.device)

        self.region_loaders = None
        if getattr(self.params, 'regions', None):
            # 多区域：样本为 (x, y, region_idx)，每个 batch 只含同一区域（长度一致，免 padding）
            # 区域均衡训练时，每个区域再单独一个 DataLoader（每步各区域各取一个 batch）
            if bool(getattr(self.params, 'region_balanced', True)):
                from utils.data_loader import get_region_dataloaders
                _, self.region_loaders = get_region_dataloaders(self.params, 'train')
                _, self.region_loaders_eval = get_region_dataloaders(self.params, 'eval')
                self.train_dataloader = get_region_dataloader(self.params, 'train')
                self.eval_dataloader = get_region_dataloader(self.params, 'eval')
            else:
                self.train_dataloader = get_region_dataloader(self.params, 'train')
                self.eval_dataloader = get_region_dataloader(self.params, 'eval')
        else:
            self.region_loaders_eval = None
            self.train_dataloader = get_dataloader(self.params, 'train')
            self.eval_dataloader = get_dataloader(self.params, 'eval')

        _llh0 = getattr(self.params, 'llh_path', None)
        if _llh0 is None:
            from utils.region_registry import region_data_paths
            _rs = list(getattr(self.params, 'regions', None) or
                       (getattr(self.params, 'region_configs', None) or {}).keys())
            _llh0 = region_data_paths(self.params, _rs[0])['llh_path']
            _mk0 = region_data_paths(self.params, _rs[0]).get('mask_key', _rs[0])
        else:
            _mk0 = getattr(self.params, 'mask_key', None)
        # 兜底用（多区域下真正生效的是下面按区域建的 self.loss_objs）
        self.loss_obj = LpLoss_weighted(
            _llh0,
            mask_path=getattr(self.params, 'mask_path', None),
            mask_key=_mk0,
            mask_boost=getattr(self.params, 'mask_boost', 1.0),
            loss_mode=getattr(self.params, 'loss_mode', 'l2'),
        )
        # 损失分量累加器（目前只有一个 'all' 分量），给 tensorboard 用
        self._loss_components = {}
        self._loss_calls = 0
        self.config = Config(self.params)
        # 多区域：每个区域一个 loss（深度加权 / mask 与区域绑定，llh 路径来自 region_layouts）
        self.loss_objs = {}
        if getattr(self.config, 'regions', None):
            for r, lay in self.config.region_layouts.items():
                self.loss_objs[r] = LpLoss_weighted(
                    lay['llh_path'],
                    mask_path=getattr(self.params, 'mask_path', None),
                    mask_key=lay.get('mask_key', r),
                    mask_boost=getattr(self.params, 'mask_boost', 1.0),
                    loss_mode=getattr(self.params, 'loss_mode', 'l2'),
                )
        self.model = SSEPREModel(self.config).to(self.device)  # params
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params.lr, weight_decay=0.01, eps=1e-6)
        # 调度器的步数必须用"真实会走的 optimizer.step() 次数"：
        #   - region_balanced 时每 epoch 只走 max(各区 batch 数)，而不是混合 loader 的总 batch 数；
        #   - 总轮数是 stage1 + stage2（两阶段训练），而不是 max_epochs。
        # 否则 warmup 会被拉长几十倍，整个训练都处在 lr≈0 的 warmup 里（之前的 bug）。
        _s1 = int(getattr(self.params, 'stage1_epochs', 0) or 0)
        _s2 = int(getattr(self.params, 'stage2_epochs', 0) or 0)
        total_epochs = (_s1 + _s2) if (_s1 > 0 and _s2 > 0) else int(self.params.max_epochs)
        per_epoch_steps = (self._region_balanced_steps_per_epoch() if getattr(self, 'region_loaders', None)
                           else len(self.train_dataloader))
        total_steps = max(1, total_epochs * max(1, per_epoch_steps))
        warmup_steps = int(total_steps * 0.05)
        print("[sched] 总轮数 %d × 每 epoch %d 步 = %d 步 | warmup %d 步 | lr=%g"
              % (total_epochs, per_epoch_steps, total_steps, warmup_steps, float(params.lr)))
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps
                                                         )
        self.writer = SummaryWriter(self.params.writer)

        # 时间窗设置
        self.window = int(getattr(params, 'time_chans', 7))
        self.block_len = int(getattr(params, 'block_len', 1))
        self.n_blocks = int(getattr(params, 'n_blocks', 1)) if params.two_step_training else 1
        self.data_chans = int(params.data_chans)
        self.horizon = self.block_len * self.n_blocks

        #设置早停
        self.patience = params.early_stopping_patience
        self.early_stop_counter = 0

        self.log_model()

        pretrained_ckpt = getattr(params, 'pretrained_ckpt_path', None)
        if pretrained_ckpt and (params.two_step_training or getattr(params, 'pretrained', False)):
            self.restore_checkpoint(pretrained_ckpt)

    # ------------------------------------------------------------------ 损失
    def _compute_loss(self, pred, target, region=None):
        """调用 loss 并累加各分量（L_all / L_mask / ...），用于分项记录。

        ``region`` 非空时用该区域自己的 loss（深度权重按区域算）。
        """
        loss_obj = self.loss_objs[region] if (region is not None and region in self.loss_objs) else self.loss_obj
        loss = loss_obj(pred, target)
        for key, value in getattr(loss_obj, 'last_components', {}).items():
            self._loss_components[key] = self._loss_components.get(key, 0.0) + value
        self._loss_calls += 1
        return loss

    def _reset_loss_components(self):
        self._loss_components = {}
        self._loss_calls = 0
        self._loss_calls_delta = 0

    def _loss_component_logs(self, prefix):
        if not self._loss_components:
            return {}
        calls = max(1, self._loss_calls)
        names = {'all': 'full', 'mask': 'mask', 'nonmask': 'nonmask'}
        out = {}
        for k, v in self._loss_components.items():
            n = max(1, getattr(self, '_loss_calls_delta', 1)) if k == 'delta' else calls
            out[f'{prefix}_loss_{names.get(k, k)}'] = v / n
        return out

    # ------------------------------------------------------------------ batch
    def _unpack_batch(self, data):
        """(x, y) 单区域 / (x, y, region_idx) 多区域 → (x, y, region_name|None)。"""
        if len(data) == 3:
            x, y, rid = data
            names = list(self.config.region_layouts.keys()) if getattr(self.config, 'region_layouts', None) else []
            region = names[int(rid[0])] if names else None
            return x, y, region
        x, y = data
        return x, y, None

    # ------------------------------------------------------------------ 前向
    def forward_blocks(self, x, y=None, region=None):
        """x: (B, W, C, L) 输入窗口；y: (B, H, C, L) 或 None。

        返回 (blocks, data_loss, aux_loss_total)
        blocks: 每个 block 的预测，元素形状 (B, block_len, C, L)
        """
        B, W, C, L = x.shape
        cur = x
        blocks = []
        data_loss = None
        aux_total = None

        for b in range(self.n_blocks):
            base = cur[:, -1]                               # 当前窗口最后一天（绝对场，归一化空间）
            out, aux = self.model(cur, region=region)       # (B, block_len*C, L)；predict_delta 时已是绝对场
            block = out.reshape(B, self.block_len, C, L)
            blocks.append(block)

            aux_total = aux if aux_total is None else aux_total + aux
            if y is not None:
                start = b * self.block_len
                target = y[:, start:start + self.block_len]
                if target.shape[1] == self.block_len:
                    loss_b = self._compute_loss(out, target.reshape(B, self.block_len * C, L), region=region)
                    # 增量项：直接监督"每天的变化"，避免模型停在"复制最后一天"
                    if (self.config.predict_delta and self.config.delta_weight > 0
                            and str(getattr(self.params, 'target_mode', 'next_day')) == 'next_day'):
                        d_pred = block - base.unsqueeze(1)              # (B, block_len, C, L)
                        d_true = target - base.unsqueeze(1)
                        loss_obj_r = (self.loss_objs[region]
                                      if (region is not None and region in self.loss_objs) else self.loss_obj)
                        # 分母用**绝对场**目标（norm=），不用 Δ 自身：
                        # 滑移日增量常常接近 0，若除以 Σw|Δ_true| 会让相对误差爆掉、梯度变噪声。
                        # 这样 L_delta ≈ "增量误差占场幅度的比例"，量纲稳定、跨区域可比。
                        # 注意用 .rel() 而不是 __call__：__call__ 会覆盖 loss_obj.last_components，
                        # 把 Δ 项写进 'all'，导致 train_loss_full 被污染（之前 full 与 delta 恒相等的原因）。
                        loss_d = loss_obj_r.rel(
                            d_pred.reshape(B, self.block_len * C, L),
                            d_true.reshape(B, self.block_len * C, L),
                            norm=target.reshape(B, self.block_len * C, L),
                        )
                        self._loss_components['delta'] = (
                            self._loss_components.get('delta', 0.0) + float(loss_d.detach()))
                        self._loss_calls_delta = getattr(self, '_loss_calls_delta', 0) + 1
                        loss_b = loss_b + self.config.delta_weight * loss_d
                    data_loss = loss_b if data_loss is None else data_loss + loss_b

            # 关键：窗口右移 block_len 天，把本段预测接在窗口尾部（丢最老的 block_len 天）。
            #   time_chans=2, block_len=1 → [第2天, 预测的第3天] → 预测第4天
            #   block_len=time_chans     → 整窗被预测替换（最初的行为）
            cur = torch.cat([cur[:, self.block_len:], block], dim=1)

        if aux_total is None:
            aux_total = torch.zeros((), device=x.device)
        return blocks, data_loss, aux_total

    def block_rmse(self, blocks, y):
        rmse = []
        for b, block in enumerate(blocks):
            start = b * self.block_len
            target = y[:, start:start + self.block_len]
            if target.shape[1] != self.block_len:
                continue
            rmse.append(torch.sqrt(torch.mean((block - target) ** 2)))
        if not rmse:
            return torch.zeros((), device=y.device)
        return torch.stack(rmse).mean()

    # ------------------------------------------------------------------ 训练
    def train(self):
        self.best_valid_loss = 1.e6
        self.stage = 1
        stage1 = int(getattr(self.config, 'stage1_epochs', 0) or 0)
        # 总 epoch：给了 stage2_epochs 就用 stage1 + stage2，否则用 max_epochs
        stage2 = int(getattr(self.config, 'stage2_epochs', 0) or 0)
        total_epochs = (stage1 + stage2) if (stage1 > 0 and stage2 > 0) else int(self.params.max_epochs)
        if total_epochs != int(self.params.max_epochs):
            print("[train] 总 epoch = stage1(%d) + stage2(%d) = %d（忽略 max_epochs=%d）"
                  % (stage1, stage2, total_epochs, int(self.params.max_epochs)))

        for epoch in range(total_epochs):
            if stage1 > 0 and self.stage == 1 and epoch >= stage1:
                self._switch_to_stage2()
            # 训练验证
            train_logs = self.train_one_epoch(epoch)
            eval_logs = self.eval_one_epoch(epoch)

            # 保存模型
            self.save_checkpoint(self.params.checkpoint_path)
            # 四区平均意义下的 best（跨阶段看 global_best.ckpt，本文件是"当前阶段最优"）
            if eval_logs['eval_loss'] < self.best_valid_loss:
                self.best_valid_loss = eval_logs['eval_loss']
                self.save_checkpoint(self.params.best_checkpoint_path)

            per_region = (self.stage == 2 and self.config.stage2_per_region_best
                          and getattr(self, 'region_best', None))
            if per_region:
                # 阶段二：每个区域只按自己的 eval loss 选 best、各自早停（heads 互不共享，独立才有意义）
                improved = []
                for r in self.region_best:
                    cur = eval_logs.get('eval_loss_' + r)
                    if cur is None:
                        continue
                    if cur < self.region_best[r]:
                        self.region_best[r] = cur
                        self.region_counter[r] = 0
                        improved.append(r)
                        self.save_checkpoint(self._region_best_path(r))
                    else:
                        self.region_counter[r] += 1
                print("[stage2] per-region best | "
                      + " ".join("%s=%.4f%s" % (r[:4], self.region_best[r],
                                                "*" if r in improved else " ")
                                for r in self.region_best)
                      + " | no-improve " + " ".join("%s:%d" % (r[:4], self.region_counter[r])
                                                    for r in self.region_counter))
                if all(c >= self.patience for c in self.region_counter.values()):
                    print(f"Early stopping at epoch {epoch+1}: 所有区域都已连续 {self.patience} epoch 无改善")
                    break
            else:
                if eval_logs['eval_loss'] < self.best_valid_loss:
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}. Validation loss hasn't improved for {self.patience} epochs.")
                        break

            # 写入tensorboard
            self.writer.add_scalar('stage', float(self.stage), epoch)
            self.writer.add_scalar('best_valid_loss', self.best_valid_loss, epoch)
            self.writer.add_scalar('train_loss', train_logs['train_loss'], epoch)
            self.writer.add_scalar('eval_loss', eval_logs['eval_loss'], epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], epoch)
            # 分项损失（train_loss_full / eval_loss_full），便于和改造前的曲线对比
            for key, value in {**train_logs, **eval_logs}.items():
                if key in ('train_loss', 'eval_loss'):
                    continue
                self.writer.add_scalar(key, value, epoch)

        print('训练结束')

    def _region_balanced_steps_per_epoch(self):
        return max(len(l) for l in self.region_loaders.values())

    def _region_balanced_train_epoch(self, epoch, train_bar=None):
        """区域均衡：每个 update 各区域各取一个 batch -> 梯度累积 -> 一次 optimizer.step()。

        每个区域的 loss 先按自己 batch 求平均，再除以区域数累加，等价于
        L = (L_A + L_B + L_C + L_D) / 4，避免样本量大的区域主导更新。
        """
        regions = list(self.region_loaders.keys())
        n_reg = len(regions)
        iters = {r: iter(self.region_loaders[r]) for r in regions}
        n_steps = self._region_balanced_steps_per_epoch()
        running = {r: 0.0 for r in regions}
        total, total_batches = 0.0, 0
        bar = train_bar if train_bar is not None else tqdm(range(n_steps))
        for step in bar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for r in regions:
                try:
                    x, y, _ = next(iters[r])
                except StopIteration:
                    iters[r] = iter(self.region_loaders[r])
                    x, y, _ = next(iters[r])
                x = x.to(self.device); y = y.to(self.device)
                blocks, data_loss, aux_loss = self.forward_blocks(x, y, region=r)
                loss_r = data_loss + aux_loss * self.config.aux_loss_coef
                (loss_r / n_reg).backward()          # 梯度累积，一次 step 等价于区域平均
                running[r] += float(loss_r.detach())
                step_loss += float(loss_r.detach())
            self.optimizer.step()
            self.scheduler.step()
            total += step_loss / n_reg
            total_batches += 1
            if hasattr(bar, "desc"):
                bar.desc = "train epoch[{}/{}] loss:{:.3f} ".format(
                    epoch + 1, self.params.max_epochs, total / max(1, total_batches)) +                     " ".join(f"{r[:4]}:{running[r] / max(1, total_batches):.3f}" for r in regions)
        logs = {'train_loss': total / max(1, total_batches)}
        for r in regions:
            logs[f'train_loss_{r}'] = running[r] / max(1, total_batches)
        logs.update(self._loss_component_logs('train'))
        return logs

    def _region_best_path(self, region):
        best = str(self.params.best_checkpoint_path)
        return os.path.join(os.path.dirname(best), 'region_%s_best.ckpt' % region)

    def _global_best_path(self):
        """阶段一的 best 另存为 global_best.ckpt（否则会被阶段二覆盖掉，无法回退）。"""
        best = str(self.params.best_checkpoint_path)
        d = os.path.dirname(best)
        name = str(getattr(self.params, 'global_best_checkpoint_path', '') or '')
        return name if name else os.path.join(d, 'global_best.ckpt')

    def _switch_to_stage2(self):
        """第二阶段：冻结共享主干，只训练区域 Adapter / Head（必要时重建优化器与调度器）。"""
        # 1) 先把阶段一的 best 留档，再重置最优判定（阶段二用阶段二自己的标准）
        gb = self._global_best_path()
        if self.best_valid_loss < 1e5:
            self.save_checkpoint(gb)
            print("[stage2] 阶段一 best (eval=%.4f) 已另存: %s" % (self.best_valid_loss, gb))
        else:
            print("[stage2] 阶段一没有产生 best（eval 未评估成功？），跳过 global_best 保存")
        self.best_valid_loss = 1.e6
        self.early_stop_counter = 0
        if int(getattr(self.config, 'stage2_patience', 0) or 0) > 0:
            self.patience = int(self.config.stage2_patience)
            print("[stage2] patience 切换为 %d" % self.patience)

        # 2) 冻结共享主干（stage2_train 决定是否连 adapter 一起训）
        keep_adapters = 'adapter' in self.config.stage2_train
        print("[stage2] 冻结共享主干，仅优化 %s" % ('各区域 head + adapter' if keep_adapters
                                                    else '各区域 head（线性探针）'))
        self.model.freeze_shared(freeze_region_embed=self.config.freeze_region_embed,
                                 keep_adapters=keep_adapters)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("stage2 没有可训练参数：请检查 region_adapter / region_head 是否为 True")

        # 3) 重建优化器 + 调度器（步数 = 剩余 epoch × **每 epoch 实际的 update 步数**）
        remaining = max(1, int(getattr(self.config, 'stage2_epochs', 0) or 0)
                        or (int(self.params.max_epochs) - int(self.config.stage1_epochs)))
        per_epoch = (self._region_balanced_steps_per_epoch() if getattr(self, 'region_loaders', None)
                     else len(self.train_dataloader))
        steps = remaining * max(1, per_epoch)
        lr2 = self.config.stage2_lr if self.config.stage2_lr > 0 else float(self.params.lr)
        self.optimizer = torch.optim.AdamW(trainable, lr=lr2, weight_decay=0.01, eps=1e-6)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(steps * 0.05), num_training_steps=steps)
        print("[stage2] 优化器重建：剩余 %d epoch × %d 步/epoch = %d 步，lr=%.1e"
              % (remaining, per_epoch, steps, lr2))
        # 分区独立选 best / 早停的状态
        regs = list(getattr(self.config, 'regions', []) or [])
        self.region_best = {r: 1.e6 for r in regs}
        self.region_counter = {r: 0 for r in regs}
        if self.config.stage2_per_region_best and regs:
            print("[stage2] 分区独立选择：每区各自的 best/早停，存 region_<r>_best.ckpt")
        self.stage = 2

    def train_one_epoch(self, epoch):
        self.model.train()
        self._reset_loss_components()
        if getattr(self, 'region_loaders', None):
            return self._region_balanced_train_epoch(epoch)
        running_loss = 0
        running_rmse = 0
        total_batches = 0
        train_bar = tqdm(self.train_dataloader)  # 使用tqdm形成trainbar
        for data in train_bar:
            x_train, y_train, region = self._unpack_batch(data)
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            blocks, data_loss, aux_loss = self.forward_blocks(x_train, y_train, region=region)
            loss = data_loss + aux_loss * self.config.aux_loss_coef
            rmse = self.block_rmse(blocks, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            running_rmse += rmse.item()
            total_batches += 1
            # 显示平均loss和rmse
            avg_loss = running_loss / total_batches
            avg_rmse = running_rmse / total_batches
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} rmse:{:.3f}".format(epoch + 1, self.params.max_epochs,
                                                                                 avg_loss, avg_rmse)

        # 返回整个epoch的平均loss（+ 各分量）
        logs = {'train_loss': running_loss / max(1, total_batches)}
        logs.update(self._loss_component_logs('train'))
        return logs

    def eval_one_epoch(self, epoch):
        self.model.eval()
        self._reset_loss_components()
        # 多区域：每个区域单独评估，报告每区域 loss + 四区平均（避免某区变差被平均掩盖）
        if getattr(self, 'region_loaders_eval', None):
            per_region = {}
            skill, persist = {}, {}
            with torch.no_grad():
                for r, loader in self.region_loaders_eval.items():
                    tot, n, sk, ps = 0.0, 0, 0.0, 0.0
                    loss_obj = self.loss_objs[r]
                    for data in loader:
                        x, y, _ = self._unpack_batch(data)
                        x, y = x.to(self.device), y.to(self.device)
                        blocks, data_loss, aux_loss = self.forward_blocks(x, y, region=r)
                        tot += float((data_loss + aux_loss * self.config.aux_loss_coef)) * x.shape[0]
                        # 纯精度项（不含 Δ 项）与 persistence 基线，同口径（相对加权 L1）
                        B = x.shape[0]
                        for b, blk in enumerate(blocks):
                            tgt = y[:, b * self.block_len:(b + 1) * self.block_len]
                            if tgt.shape[1] != self.block_len:
                                continue
                            t_ = tgt.reshape(B, -1, tgt.shape[-1])
                            sk += float(loss_obj.rel(blk.reshape(B, -1, blk.shape[-1]), t_)) * B
                            base = x[:, -1].unsqueeze(1).expand(-1, self.block_len, -1, -1)
                            ps += float(loss_obj.rel(base.reshape(B, -1, base.shape[-1]), t_)) * B
                        n += B
                    nb = max(1, len(blocks)) * max(1, n)
                    per_region[r] = tot / max(1, n)
                    skill[r] = sk / nb
                    persist[r] = ps / nb
            mean_loss = sum(per_region.values()) / max(1, len(per_region))
            print("[eval] " + " | ".join(f"{r[:4]}={v:.4f}" for r, v in per_region.items())
                  + f" | mean={mean_loss:.4f}")
            # 真正的"学得好不好"：纯精度项 vs persistence（同一相对 L1 口径）
            print("[eval-skill] " + " | ".join(
                f"{r[:4]}={skill[r]:.4f}(persist {persist[r]:.4f}, 增益 "
                f"{100.0 * (1 - skill[r] / max(persist[r], 1e-9)):+.1f}%)" for r in skill))
            logs = {'eval_loss': mean_loss}
            for r, v in per_region.items():
                logs[f'eval_loss_{r}'] = v
            for r in skill:
                logs[f'eval_skill_{r}'] = skill[r]
                logs[f'eval_persist_{r}'] = persist[r]
                logs[f'eval_gain_{r}'] = 100.0 * (1 - skill[r] / max(persist[r], 1e-9))
            if skill:
                logs['eval_gain_mean'] = sum(100.0 * (1 - skill[r] / max(persist[r], 1e-9))
                                             for r in skill) / len(skill)
            return logs
        running_loss = 0
        running_rmse = 0
        total_batches = 0
        eval_bar = tqdm(self.eval_dataloader)

        for data in eval_bar:
            x_train, y_train, region = self._unpack_batch(data)
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            with torch.no_grad():
                blocks, data_loss, aux_loss = self.forward_blocks(x_train, y_train, region=region)
                loss = data_loss + aux_loss * self.config.aux_loss_coef
                rmse = self.block_rmse(blocks, y_train)

            running_loss += loss.item()
            running_rmse += rmse.item()
            total_batches += 1
            avg_loss = running_loss / total_batches
            avg_rmse = running_rmse / total_batches
            eval_bar.desc = "eval epoch[{}/{}] loss:{:.3f} rmse:{:.3f}".format(epoch + 1, self.params.max_epochs,
                                                                               avg_loss, avg_rmse)

        logs = {'eval_loss': running_loss / max(1, total_batches)}
        logs.update(self._loss_component_logs('eval'))
        return logs

    def save_checkpoint(self, checkpoint_path, model=None):
        """ 保存模型"""
        if not model:
            model = self.model

        d = os.path.dirname(str(checkpoint_path))
        if d:
            os.makedirs(d, exist_ok=True)      # 新 run（如 multiregion/00）目录可能还不存在
        # 一并存下会改变"输出语义"的开关：predict_delta 打开后，ckpt 只能配 predict_delta=true 用，
        # 否则推理会多加一次 base（旧 ckpt 用新配置评估会静默算错）。加载时据此告警。
        meta = {'predict_delta': bool(getattr(self.config, 'predict_delta', False)),
                'patch_size': int(getattr(self.config, 'patch_size', 0)),
                'time_chans': int(getattr(self.config, 'time_chans', 0)),
                'block_len': int(getattr(self.config, 'block_len', 0)),
                'n_blocks': int(getattr(self.config, 'n_blocks', 0)),
                'pos_embed_mode': str(getattr(self.config, 'pos_embed_mode', '')),
                'regions': list(getattr(self.config, 'regions', []) or [])}
        torch.save({'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                    'meta': meta}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """ 读取 checkpoint。

        输入窗长度/通道数变化时 patch_embed 与 head_out 的形状会变：
        - patch_embed: 按"每天 old_data_chans 个通道 -> 每天 new_data_chans 个通道"逐个时间步复制，
          新增的时间步(例如从 2 天窗口扩到 7 天)与新增通道用 kaiming/随机初始化补齐；
        - head_out: 直接丢弃不加载。
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"pretrained_ckpt_path 不存在: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state']

        # 语义开关不一致时告警（不会自动改配置，避免静默算出错误结果）
        meta = checkpoint.get('meta') or {}
        for k in ('predict_delta', 'patch_size', 'time_chans', 'block_len', 'n_blocks', 'pos_embed_mode'):
            if k in meta and hasattr(self.config, k) and str(getattr(self.config, k)) != str(meta[k]):
                print(f"[restore_checkpoint] ⚠ 配置 {k}={getattr(self.config, k)!r} 与 ckpt 记录的 "
                      f"{meta[k]!r} 不一致；predict_delta 不一致会导致输出多加/少加一次 base")

        # --- 修正 Patch Embedding 的权重加载 ---
        embed_key = 'patch_embed.proj.weight'
        if embed_key in state_dict:
            old_weight = state_dict[embed_key]                # [Hidden, Old_Time*Old_Data, Patch]
            new_weight = self.model.patch_embed.proj.weight   # [Hidden, New_Time*New_Data, Patch]

            if old_weight.shape != new_weight.shape:
                time_steps = self.config.time_chans            # 新的窗口天数
                old_in = old_weight.shape[1]
                new_in = new_weight.shape[1]
                old_data_chans = old_in // time_steps if old_in % time_steps == 0 else old_in
                new_data_chans = new_in // time_steps

                with torch.no_grad():
                    nn.init.kaiming_normal_(self.model.patch_embed.proj.weight)
                    old_steps = min(time_steps, old_in // max(1, old_data_chans))
                    for t in range(old_steps):
                        old_start = t * old_data_chans
                        old_end = old_start + old_data_chans
                        new_start = t * new_data_chans
                        new_end = new_start + min(old_data_chans, new_data_chans)
                        self.model.patch_embed.proj.weight[:, new_start:new_end, :] = \
                            old_weight[:, old_start:old_end, :][:, :new_end - new_start, :]
                del state_dict[embed_key]
                print(f"[restore] patch_embed 形状不同 {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}，"
                      f"已按时间步复制旧权重，其余随机初始化")

        # --- 移除不匹配的 Head 层 ---
        head_keys = ['head_out.weight', 'head_out.bias']
        for key in head_keys:
            if key in state_dict:
                del state_dict[key]

        # --- 丢弃形状不匹配的纯缓存张量（如 RoPE 的 cos/sin 表：老 checkpoint 是 1113 长度，
        #     当前 align_tokens 后是 1692，形状不同 load_state_dict 会直接报 size mismatch）---
        cur = self.model.state_dict()
        drop = [k for k, v in state_dict.items() if k in cur and cur[k].shape != v.shape]
        for k in drop:
            del state_dict[k]
        if drop:
            print(f"[restore] 丢弃 {len(drop)} 个形状不匹配的张量（会重新初始化）: "
                  f"{drop[:3]}{' ...' if len(drop) > 3 else ''}")

        # --- 加载其余匹配的层 ---
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[restore] loaded {checkpoint_path} | missing={len(missing)} unexpected={len(unexpected)}")

    def log_model(self):
        """记录model到tensorboard（多区域时跳过：各区域 token 数不同，单张图无法覆盖）"""
        if getattr(self.config, 'regions', None):
            print("[log_model] 多区域模式跳过 add_graph（各区域 fea_len/token 数不同）")
            return
        init_img = torch.zeros((1, self.window, self.data_chans,
                                int(getattr(self, 'fea_len', getattr(self.params, 'img_size', 0)))),
                               device=self.device)
        self.writer.add_graph(self.model, init_img)


def _acquire_run_lock(exper_path):
    """同一个 exper_path 只允许一个训练进程（防止误开多个 run 抢写同一份 ckpt/tensorboard）。"""
    path = os.path.join(str(exper_path), '.train.lock')
    os.makedirs(str(exper_path), exist_ok=True)
    if os.path.isfile(path):
        try:
            pid = int(open(path).read().strip() or 0)
            os.kill(pid, 0)                      # 进程还活着 → 拒绝
            raise SystemExit(
                "[lock] %s 已有训练进程 pid=%d 在跑（锁文件 %s）。\n"
                "       同一份 exper_path 不能同时跑两个 run；请先结束它，或换个 exper_path。"
                % (exper_path, pid, path))
        except (ValueError, ProcessLookupError, PermissionError):
            pass                                  # 陈旧锁 → 覆盖
    with open(path, 'w') as f:
        f.write(str(os.getpid()))
    import atexit
    atexit.register(lambda: os.path.isfile(path) and os.remove(path))


if __name__ == '__main__':
    seed_everything(111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', default='./config/multiregion_l2_ddp.yaml', type=str)
    parser.add_argument("--config", default='backbone', type=str)  # 配置的部分
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    _acquire_run_lock(getattr(params, 'exper_path', './Results'))

    trainer = Trainer(params)
    trainer.train()
