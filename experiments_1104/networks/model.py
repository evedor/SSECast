import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from utils.compat import DropPath, trunc_normal_, PretrainedConfig
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1113):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)


def repeat_kv(hidden_states, n_rep):
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.dropout = config.dropout  # config.dropout
        self.hidden_size = hidden_size  # config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_cache, self.v_cache = None, None
        # self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        # RoPE：1-D（按 token 序号），表长度必须跟 token 数一致（align_tokens 会把 1113 变成 1692）
        self.rotary_emb = RotaryEmbedding(self.head_dim, int(getattr(config, 'max_seq_len', 1113)))
        self.b = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

    def forward(self, hidden_states, use_kv_cache=False):
        b, s = hidden_states.shape[:2]
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_key_value_heads, self.head_dim)
        v = v.view(b, s, self.num_key_value_heads, self.head_dim)

        q, k = self.rotary_emb(q, k)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q = q.transpose(1, 2)  # b, self.num_heads, s, self.head_dim
        k = k.transpose(1, 2)  # b, self.num_heads, s, self.head_dim
        v = v.transpose(1, 2)  # b, self.num_heads, s, self.head_dim

        # 用 PyTorch 融合的 SDPA（H200 上会自动选 flash kernel）：
        #   - 不再显式生成 (b, heads, s, s) 的注意力矩阵（batch=48、s=1113 时单层就是 ~3.8 GB float32），
        #     显存从"每层常驻大矩阵"降到 O(b*s*d)，速度也快很多；
        #   - dropout_p 作用在注意力权重上，与原先对 scores 做 dropout 语义一致；
        #   - self.b 是形状 (1, heads, 1, 1) 的逐头常数偏置，被广播到整个 s×s 上，
        #     而 softmax(s + c) == softmax(s)，数学上没有作用，这里不再显式相加
        #     （参数保留以兼容已有 state_dict）。
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        output = output.transpose(1, 2).contiguous().view(b, s, -1)  # b, s, self.hidden_size

        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output


class Gating(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num)

    def forward(self, x):
        # x dim: b, s, hidden_size
        logits = self.gate(x)  # gate: b, s, expert_num
        logits_topk, indices = logits.topk(self.topk, dim=-1)  # 选择概率最大的两个专家，返回两个专家对每个token的概率
        zeros = torch.full_like(logits, float("-inf"))  # 创建一个全为负无穷的矩阵，用于屏蔽其他专家的概率并重新归一化概率最大的两个专家
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # 将选择的两个专家的概率按指定索引填充
        sparse_logits = F.softmax(sparse_logits, dim=-1)  # 得到一个稀疏矩阵，选择的两个专家对每个token的概率和为1
        gate_logit = logits.view(-1, self.expert_num)

        return sparse_logits, indices, gate_logit


class Expert(nn.Module):
    def __init__(self, config,hidden_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        down_proj = self.down_proj(x)
        return down_proj

class SharedExpert(nn.Module):
    """共享专家：捕捉全局普适的物理规律"""
    def __init__(self, config, hidden_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.w2 = nn.Linear(config.intermediate_size, hidden_size, bias=config.mlp_bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class MoE(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList([Expert(config, self.hidden_size) for _ in range(config.expert_num)])
        self.shared_expert = SharedExpert(config, self.hidden_size)
        self.gating = Gating(config, self.hidden_size)

    def forward(self, x):
        shared_output = self.shared_expert(x)
        sparse_logits, indices, gate_logit = self.gating(x)
        final_outputs = torch.zeros_like(x)
        x_flat = x.reshape(-1, x.shape[-1])  # (batch_size * seq_len, dim)
        sparse_logits_flat = sparse_logits.reshape(-1, sparse_logits.shape[-1])  # (batch_size * seq_len, export_num))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)  # (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1)  # (batch_size * seq_len)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # (seq_true, dim)
                export_output = expert(expert_input)  # (seq_true, dim)

                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true) --> (seq_true, 1)

                weighted_output = export_output * gate_scores  # (seq_true, dim)

                final_outputs[expert_mask] += weighted_output

        return final_outputs + shared_output, gate_logit


class MLP(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        down_proj = self.down_proj(x)
        return down_proj


class BasicLayer(nn.Module):
    def __init__(self, config, hidden_size, layer_idx):
        super().__init__()
        self.dim = hidden_size
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(self.dim)

        self.self_attn = Attention(config, self.dim)

        self.drop_path = DropPath(config.dropout) if config.dropout > 0. else nn.Identity()
        self.norm2 = RMSNorm(self.dim)
        #第一层用mlp，第二层用moe
        self.mlp = MLP(config, self.dim)
        self.moe = MoE(config, self.dim)

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        residual = x
        x = self.norm1(x)

        # Self Attention
        x = self.self_attn(x)

        x = residual + self.drop_path(x)
        # Fully Connected
        residual = x
        x = self.norm2(x)

        if self.layer_idx % 2 == 0:
            x = self.mlp(x)
            gate_logit = None
        else:
            x, gate_logit = self.moe(x)

        outputs = residual + self.drop_path(x)
        return outputs, gate_logit

class Block(nn.Module):
    def __init__(self, config, hidden_size, depth):
        super(Block, self).__init__()
        self.config = config
        self.depth = depth
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(depth):
            self.layers.append(BasicLayer(config, hidden_size, layer_idx))

    def forward(self,x):
        for idx, layer in enumerate(self.layers):
            x, gate_logit = layer(x)
        return x, gate_logit

class UpSample(nn.Module):
    def __init__(self, in_dim, in_length, out_length):
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim // 2, bias=False)  # Mimic DownSample1D: in_dim -> in_dim/2
        self.norm = nn.LayerNorm(in_dim)  # Normalize over input features
        self.in_length = in_length
        self.out_length = out_length
        # Padding to align output length if needed
        self.pad = nn.ConstantPad1d((0, out_length - in_length * 2), 0) if out_length > in_length * 2 else None

    def forward(self, x, out_length=None):
        B, N, C = x.shape  # x: (batch_size, in_length, in_dim)
        # out_length 由调用方传入（= skip 的长度）。多区域训练时各区域 token 数不同，
        # 不能再依赖构造期固定的 self.out_length。
        target = int(out_length) if out_length is not None else int(self.out_length)
        x = x.transpose(1, 2)  # Shape: (B, C, in_length)
        x = F.interpolate(x, size=target, mode='linear', align_corners=False)  # Shape: (B, C, target)
        x = x.transpose(1, 2)  # Shape: (B, target, C)
        # if self.pad is not None:
        #     x = self.pad(x)  # Shape: (B, out_length, C)
        x = self.norm(x)  # Shape: (B, target, C)
        x = self.linear(x)  # Shape: (B, target, C // 2)

        return x

class DownSample(nn.Module):
    def __init__(self, in_dim=768, in_length=1113, out_length=557):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, in_dim * 2, bias=False)  # Mimic Pangu-Weather: 4*in_dim -> 2*in_dim
        self.norm = nn.LayerNorm(2 * in_dim)  # Normalize over grouped features
        self.in_length = in_length
        self.out_length = out_length

        # Calculate padding to align dimensions
        pad_size = out_length * 2 - in_length
        self.pad = nn.ConstantPad1d((0, pad_size), 0)  # Pad on the right

    def forward(self, x):
        B, N, C = x.shape  # x: (batch_size, in_length, in_dim)
        # 下采样长度按实际 token 数算（out = ceil(N/2)），不再依赖构造期的 in_length/out_length
        out = (N + 1) // 2
        if N % 2:                       # 奇数长度：末尾复制一位，保证能两两配对
            x = torch.cat([x, x[:, -1:]], dim=1)
            N += 1
        x = x.reshape(B, out, 2, C).reshape(B, out, 2 * C)  # (B, out, 2C)
        x = self.norm(x)
        x = self.linear(x)  # Shape: (B, out, 2 * C)
        return x

def build_pos_grid_coords(config, num_patches):
    """构造可学习二维网格 PE 的采样坐标。

    支持两种坐标体系（``pos_coord_mode``）：
      * ``"uv"``     —— (u, v)：u=条带内沿倾角弧长（单位≈中位 token 间距），v=沿走向条带号。
                        网格形状 (pos_grid_v, pos_grid_u)，v 方向通常正好落在整数行上（不插值）。
      * ``"lonlat"`` —— (lon, lat)：token 的**平均经纬度**。
                        网格形状 (pos_grid_lat, pos_grid_lon)，两个方向都真正双线性插值；
                        两个轴各自 min-max 归一化，(H, W) 应取成与物理长宽比（含 cos(lat)）匹配，
                        这样网格单元近似为正方形（cascadia 514×1242 km → 48×24）。

    返回 (coords, grid_h, grid_w, axis_labels)，coords 形状 (1, 1, S, 2)，值域 [-1, 1]，
    顺序是 (x=列方向, y=行方向)，与 ``F.grid_sample`` 的约定一致。
    """
    mode = str(getattr(config, 'pos_coord_mode', 'uv')).lower()
    if mode == 'lonlat':
        tab = getattr(config, 'token_lonlat', None)
        if tab is None:
            raise ValueError("pos_coord_mode='lonlat' 需要 token_lonlat（请先 align_tokens=True）")
        xy = torch.as_tensor(tab, dtype=torch.float32)          # (S, 2) = (lon, lat)
        x, y = xy[:, 0], xy[:, 1]
        # 两个轴各自 min-max 归一化到 [-1, 1]（grid_sample 的约定），因此网格单元的物理尺寸是
        #   lon 向 ≈ lon_span·cos(lat)/ (W-1)，lat 向 ≈ lat_span / (H-1)
        # 想要"方格"就按物理长宽比选 (H, W)：cascadia 是 ~514 km(经) × 1242 km(纬)，
        # 所以默认 48×24（单元 ≈ 22 km × 26 km，近似正方形）。
        grid_h = int(getattr(config, 'pos_grid_lat', 48))        # 行 = 纬度
        grid_w = int(getattr(config, 'pos_grid_lon', 24))        # 列 = 经度
        labels = ('lon', 'lat')
    elif mode == 'region':
        # 多区域：真实坐标按 region 从 config.region_layouts 取（见 SSEPREModel），这里只给占位表
        grid_h = int(getattr(config, 'pos_grid_lat', 48))
        grid_w = int(getattr(config, 'pos_grid_lon', 48))
        return torch.zeros(1, 1, num_patches, 2), grid_h, grid_w, ('lon', 'lat')
    elif mode == 'uv':
        tab = getattr(config, 'token_uv', None)
        if tab is None:
            raise ValueError("pos_coord_mode='uv' 需要 token_uv（请先 align_tokens=True）")
        uv = torch.as_tensor(tab, dtype=torch.float32)            # (S, 2) = (u, v)
        x, y = uv[:, 0], uv[:, 1]
        grid_h = int(getattr(config, 'pos_grid_v', 72))
        grid_w = int(getattr(config, 'pos_grid_u', 24))
        labels = ('u', 'v')
    else:
        raise ValueError(f"pos_coord_mode 只能是 'uv' / 'lonlat'，收到 {mode!r}")

    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    coords = torch.stack([x * 2 - 1, y * 2 - 1], dim=-1).view(1, 1, num_patches, 2)
    return coords, grid_h, grid_w, labels


class SSEPREModel(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches
        # ---- 位置编码：table（每个 token 一个可学习向量）或 grid（二维几何网格 + 双线性插值）----
        self.pos_embed_mode = str(getattr(config, 'pos_embed_mode', 'table')).lower()
        self.pos_embed = None
        self.pos_embed_grid = None
        if self.pos_embed_mode == 'grid':
            # 坐标可以是 (u,v) 几何坐标，或 token 的平均经纬度 (lon,lat) —— 见 build_pos_grid_coords
            coords, self.pos_grid_v, self.pos_grid_u, self.pos_coord_labels = \
                build_pos_grid_coords(config, num_patches)
            self.pos_coord_mode = str(getattr(config, 'pos_coord_mode', 'uv')).lower()
            self.pos_embed_grid = nn.Parameter(
                torch.zeros(1, 1, self.pos_grid_v, self.pos_grid_u, config.hidden_size))
            trunc_normal_(self.pos_embed_grid, std=.02)
            # grid_sample 的坐标是 (x=列, y=行)，范围 [-1, 1]
            self.register_buffer('pos_grid_coords', coords, persistent=False)
        elif self.pos_embed_mode in ('none', 'off', 'no', 'null'):
            pass                      # 不用任何显式位置编码：位置信息只由 RoPE（序号）+ region_embed 提供
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        # 条带对齐时，输出从 (n_slots) 收回 fea_len 个单元
        unpad_index = getattr(config, 'token_unpad_index', None)
        self.register_buffer(
            'unpad_index',
            None if unpad_index is None else torch.as_tensor(unpad_index, dtype=torch.long),
            persistent=False,
        )
        # ---- 多区域：每区域一套 回填索引 + 网格采样坐标；region embedding 表达"哪个区域"----
        region_layouts = getattr(config, 'region_layouts', None) or {}
        self.region_names = list(region_layouts.keys())
        self.region_index = {r: i for i, r in enumerate(self.region_names)}
        self._region_unpad_buffers = {}
        self._region_coords_buffers = {}
        for r, lay in region_layouts.items():
            un = 'unpad__' + str(r)
            self.register_buffer(un, torch.as_tensor(lay['unpad_index'], dtype=torch.long), persistent=False)
            self._region_unpad_buffers[r] = un
            if self.pos_embed_grid is not None:      # 坐标只在 grid 位置编码下才需要
                co = 'coords__' + str(r)
                self.register_buffer(co, torch.as_tensor(lay['grid_coords'], dtype=torch.float32),
                                     persistent=False)
                self._region_coords_buffers[r] = co
        self.region_embed = None
        if self.region_names and bool(getattr(config, 'region_embed', True)):
            n_reg = len(self.region_names)
            self.region_embed = nn.Embedding(n_reg, config.hidden_size)
            # 区域身份的初始化：onehot = 每区一个独立的 0.02 基向量（第一步就带"我是谁"的信号）；
            # zeros = 原行为（起点不干扰，但要靠梯度慢慢分化）；normal = 小随机。
            self.region_embed_init = str(getattr(config, 'region_embed_init', 'onehot')).lower()
            scale = float(getattr(config, 'region_embed_scale', 0.02))
            with torch.no_grad():
                if self.region_embed_init == 'onehot':
                    self.region_embed.weight.copy_(torch.eye(n_reg, config.hidden_size) * scale)
                elif self.region_embed_init in ('normal', 'randn'):
                    nn.init.normal_(self.region_embed.weight, std=scale)
                else:                                   # zeros（原行为）
                    nn.init.zeros_(self.region_embed.weight)

        # ---- 区域专属 Adapter（共享主干之后再插一个小残差模块，零初始化 = 起点等价于共享模型）----
        self.region_adapters = None
        if self.region_names and bool(getattr(config, 'region_adapter', False)):
            hidden = config.hidden_size
            bottleneck = max(8, hidden // int(getattr(config, 'region_adapter_ratio', 4)))
            self.region_adapters = nn.ModuleDict()
            for r in self.region_names:
                seq = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, bottleneck),
                    nn.GELU(),
                    nn.Linear(bottleneck, hidden),
                )
                nn.init.zeros_(seq[-1].weight)          # 零初始化：训练开始时 = 纯共享主干
                nn.init.zeros_(seq[-1].bias)
                self.region_adapters[r] = seq

        # ---- 区域专属 Head（最终投影按区域分开；head1/head2 仍共享）----
        self.region_heads = None
        if self.region_names and bool(getattr(config, 'region_head', False)):
            self.region_heads = nn.ModuleDict({
                r: nn.Linear(64, config.out_chans * config.patch_size, bias=True)
                for r in self.region_names
            })
            # predict_delta 时零初始化（= 该区域起点精确等于 persistence）；
            # 否则 xavier（避免输出恒 0 让主干在最初几步拿不到梯度）。
            for m in self.region_heads.values():
                if bool(getattr(config, 'predict_delta', False)):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # 已弃用：patch_embed 之后的 dropout 已移除（见 forward）。Dropout 无参数，
        # 保留这个模块不会影响任何 ckpt 的 state_dict 兼容性。
        self.pos_drop = nn.Dropout(p=config.dropout)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.layer1 = Block(config, config.hidden_size, 2)
        self.layer2 = Block(config, config.hidden_size*2, 2)
        self.layer3 = Block(config, config.hidden_size*2, 2)
        self.layer4 = Block(config, config.hidden_size, 2)
        self.layer5 = Block(config, config.hidden_size, 2)
        
        self.seqlen = int(self.config.max_seq_len)
        self.halflen = int((self.config.max_seq_len // 2) + (self.config.max_seq_len // 2 % 2))

        self.downsample = DownSample(in_dim=config.hidden_size, in_length=self.seqlen, out_length=self.halflen)
        self.upsample = UpSample(in_dim=config.hidden_size*2, in_length=self.halflen, out_length=self.seqlen)
        

        self.head1 = nn.Linear(config.hidden_size, 256, bias=True)
        self.head2 = nn.Linear(256, 64, bias=True)
        self.head_out = nn.Linear(64, config.out_chans * config.patch_size, bias=True)
        if bool(getattr(config, 'predict_delta', False)):
            # 增量模式：输出头零初始化 → 训练起点精确等于 persistence（Δ=0）。
            # 这是让"模型最差也不该比 persistence 差"结构成立的关键一步。
            nn.init.zeros_(self.head_out.weight)
            nn.init.zeros_(self.head_out.bias)
        
        self.act = nn.GELU() 
        self.head_drop = nn.Dropout(config.dropout)


        # self.load_balancing_loss_func = load_balancing_loss_func()

    # ---------------------------------------------------------------- 两阶段训练
    def trainable_parameter_groups(self, stage=1, freeze_region_embed=True, keep_adapters=True):
        """返回 (共享参数名列表, 区域专属参数名列表)，供冻结/重建优化器使用。

        ``stage=2`` 时只保留区域专属 Adapter / Head（以及可选的 region_embed）可训练，
        共享主干（patch_embed / pos_embed / 各层 / downsample / head1 / head2）全部冻结。
        """
        region_prefixes = []
        if self.region_adapters is not None and keep_adapters:
            region_prefixes.append('region_adapters.')
        if self.region_heads is not None:
            region_prefixes.append('region_heads.')
        if self.region_embed is not None and not freeze_region_embed:
            region_prefixes.append('region_embed.')
        shared, specific = [], []
        for name, _ in self.named_parameters():
            (specific if any(name.startswith(p) for p in region_prefixes) else shared).append(name)
        return shared, specific

    def freeze_shared(self, freeze_region_embed=True, keep_adapters=True):
        """第二阶段：冻结共享主干（含共享 PE / head1 / head2），只留区域模块可训练。

        ``keep_adapters=False`` → 连 region adapter 一起冻结，只训各区域的 head
        （= 每个区域一个线性探针，390 参数/区）。
        """
        shared, specific = self.trainable_parameter_groups(stage=2, freeze_region_embed=freeze_region_embed,
                                                          keep_adapters=keep_adapters)
        for name, p in self.named_parameters():
            p.requires_grad_(name in specific)
        n_shared = sum(p.numel() for n, p in self.named_parameters() if n in set(shared))
        n_spec = sum(p.numel() for n, p in self.named_parameters() if n in set(specific))
        print(f"[freeze_shared] 冻结 {len(shared)} 个共享张量（{n_shared/1e6:.2f}M 参数），"
              f"保留 {len(specific)} 个区域专属张量（{n_spec/1e6:.3f}M 参数）可训练")
        return shared, specific

    def load_balancing_loss_func(self, gate_logits,num_experts,top_k):
        concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits],
                                             dim=0)  # 各个层的gate_logit进行合并[layers X batch_size X sequence_length, num_experts]
        routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        router_prob_per_expert = torch.mean(routing_weights, dim=0)
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss * num_experts

    def forward(self, x, region=None):
        """x: (B, time_chans, data_chans, fea_len)。

        ``region`` 只在多区域训练时给出（如 ``"cascadia"``）：用来选该区域的槽位映射、
        回填索引与网格采样坐标，并加上该区域的 region embedding。单区域时保持原行为。
        """
        all_router_logits = [] if self.config.output_router_logits else None
        # 增量预测模式：模型只输出 Δ，绝对场 = 输入窗最后一天 + Δ。
        # 基准在这里取（x 还是原始 (B, time_chans, C, fea_len)），下游拿到的仍然是绝对场，
        # 所以 Trainer / notebook / 可视化都不用改。
        base = x[:, -1] if bool(getattr(self.config, 'predict_delta', False)) else None

        use_region = region if (region is not None and region in self.region_index) else None

        x = self.patch_embed(x, region=use_region)
        if self.pos_embed_grid is not None:
            # (1, C, H, W) 双线性采样到每个 token 的坐标（经纬度或 u/v）—— 位置编码在几何上是平滑的
            coords = self.pos_grid_coords if use_region is None else self._buffers[self._region_coords_buffers[use_region]]
            if coords.shape[2] != x.shape[1]:
                raise ValueError(
                    "位置编码坐标长度 %d 与 token 数 %d 不一致（region=%r）："
                    "多区域请确认传入了正确的 region，或检查 region_layouts 与输入 fea_len 是否匹配"
                    % (coords.shape[2], x.shape[1], region))
            grid_pe = F.grid_sample(
                self.pos_embed_grid.reshape(1, self.config.hidden_size, self.pos_grid_v, self.pos_grid_u),
                coords, mode="bilinear", align_corners=True,
            ).reshape(1, self.config.hidden_size, -1).transpose(1, 2)
            x = x + grid_pe
        elif self.pos_embed is not None:
            x = x + self.pos_embed
        # 区域身份向量：独立于位置编码（pos_embed_mode="none" 时也要生效）
        if self.region_embed is not None and use_region is not None:
            x = x + self.region_embed.weight[self.region_index[use_region]].view(1, 1, -1)
        
        
        x, router_logits_1 = self.layer1(x)
        skip = x

        x = self.downsample(x)
        x, router_logits_2 = self.layer2(x)
        x, router_logits_3 = self.layer3(x)
        x = self.upsample(x, out_length=skip.shape[1])   # 长度跟随实际 token 数（多区域）
        x = x + skip
        x, router_logits_4 = self.layer4(x)
        # x, router_logits_5 = self.layer5(x)

        # 区域专属 Adapter（残差形式，只作用于当前区域的 token）
        if self.region_adapters is not None and use_region is not None:
            x = x + self.region_adapters[use_region](x)

        # x = torch.concat([x, skip], dim=-1)

        x = self.head1(x)
        x = self.act(x)      
        x = self.head_drop(x)
        
        x = self.head2(x)
        x = self.act(x)       
        x = self.head_drop(x) 
        
        if self.region_heads is not None and use_region is not None:
            x = self.region_heads[use_region](x)      # 区域专属输出头
        else:
            x = self.head_out(x)
        x = rearrange(
            x,
            "b l (p c_out) -> b c_out (l p)",
            p=self.config.patch_size,
            l=x.shape[1],          # 用实际 token 数（多区域各区域不同）
        )
        unpad = self.unpad_index
        if use_region is not None:
            unpad = self._buffers[self._region_unpad_buffers[use_region]]
        if unpad is not None:
            # 去掉补齐槽位，回到 fea_len 个断层单元的顺序（下游 loss/可视化无需改动）
            x = x.index_select(-1, unpad)
        if base is not None:
            # Δ -> 绝对场：加回窗口最后一天（形状 (B, out_chans, fea_len)）
            x = x + base
        if self.config.output_router_logits:
            all_router_logits.extend([router_logits_1, router_logits_2, router_logits_3, router_logits_4]) #

        aux_loss = None
        if self.config.output_router_logits:
            aux_loss = self.load_balancing_loss_func(all_router_logits, self.config.expert_num, self.config.topk)

        return x, aux_loss


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fea_len = config.fea_len
        self.patch_size = config.patch_size
        # token 数：默认 fea_len // patch_size；开启"条带对齐"(align_tokens) 时用补齐后的 token 数
        num_patches = int(getattr(config, 'n_tokens', self.fea_len // self.patch_size))
        self.num_patches = num_patches
        self.proj = nn.Conv1d(config.time_chans * config.data_chans, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.gelu = nn.GELU()
        # 条带对齐时的槽位映射：(n_slots,) 的长整型索引，把 fea_len 个单元展开成 n_slots 个 patch 槽位
        slot_index = getattr(config, 'token_slot_index', None)
        self.register_buffer(
            'slot_index',
            None if slot_index is None else torch.as_tensor(slot_index, dtype=torch.long),
            persistent=False,
        )
        # ---- 多区域：每个区域一套槽位映射（proj / 主干 / head 全区域共享）----
        region_layouts = getattr(config, 'region_layouts', None) or {}
        self.region_slot_buffers = {}
        for r, lay in region_layouts.items():
            name = 'slot__' + str(r)
            self.register_buffer(name, torch.as_tensor(lay['slot_index'], dtype=torch.long), persistent=False)
            self.region_slot_buffers[r] = name

    def forward(self, x, region=None):
        B, D, C, L = x.shape #1,2,3,3339
        x = x.view(B, D * C, L)
        # assert L == self.fea_len, f"Input signal length ({L}) doesn't match model ({self.fea_len})."
        slot = self.slot_index
        if region is not None and region in self.region_slot_buffers:
            slot = self._buffers[self.region_slot_buffers[region]]
        if slot is not None:
            # 按条带对齐把单元序列重排/补齐（补齐位是条带最后一个单元的复制），
            # 这样每个 token 的 patch_size 个单元都落在同一个条带内
            x = x.index_select(-1, slot)
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x



class Config(PretrainedConfig):
    def __init__(self,
                 fea_len=3339,
                 time_chans=2,
                 data_chans=3,
                 out_chans=3,
                 patch_size=3,
                 hidden_size=768,
                 num_attention_heads=16,
                 num_key_value_heads=8,
                 flash_attn=True,
                 attention_bias=False,
                 max_seq_len=1113,
                 intermediate_size=1024,
                 mlp_bias=False,
                 vocab_size=6400,
                 n_layers=8,
                 dropout=0.0,
                 expert_num=4,
                 topk=2,
                 output_router_logits=True,
                 aux_loss_coef=0.01,
                 mlp_ratio=4,
                 **kwargs):
        self.fea_len = fea_len
        self.time_chans = time_chans
        self.data_chans = data_chans
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        self.mlp_ratio = mlp_ratio
        super().__init__(**kwargs)

# if __name__ == "__main__":
    # model = SSEPREModel(fea_len=3339, patch_size=3, data_chans=3, out_chans=1)
    # sample = torch.randn(1, 2, 3, 3339)
    # result = model(sample)
    # print(result.shape)
    # print(torch.norm(result))

    # config = Config()
    # model = SSEPREModel(config)
    # sample = torch.randn(1, 2, 3, 3339)#torch.randn(1, 1113, 768)
    # result,result1 = model(sample)
    # print(result.shape)
    # print(result1.shape)
    # print(torch.norm(result))