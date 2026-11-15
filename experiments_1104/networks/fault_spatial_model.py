"""SSECast with one fault element per token and geometry-aware multiscale layers."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from utils.fault_geometry import load_geometry, joint_depth_weights

MODEL_VERSION = "fault_spatial_v5_zero_init_geometry_upsample"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return norm.to(x.dtype) * self.weight.to(x.dtype)


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        mask = (torch.rand((len(x),) + (1,) * (x.ndim - 1), device=x.device) < keep).to(
            x.dtype
        )
        return x * mask / keep


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, length):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE requires even head dimension")
        freq = (
            torch.arange(int(length)).float()[:, None]
            * (10000.0 ** (-torch.arange(0, dim, 2).float() / dim))[None, :]
        )
        freq = torch.cat((freq, freq), -1)
        self.register_buffer("cos_cached", freq.cos(), persistent=False)
        self.register_buffer("sin_cached", freq.sin(), persistent=False)

    def forward(self, q, k):
        n = q.shape[1]
        if n > len(self.cos_cached):
            raise ValueError("Sequence exceeds configured RoPE length")
        cos = self.cos_cached[:n][None, :, None, :].to(q.dtype)
        sin = self.sin_cached[:n][None, :, None, :].to(q.dtype)

        def rotate(x):
            a, b = x.chunk(2, -1)
            return x * cos + torch.cat((-b, a), -1) * sin

        return rotate(q), rotate(k)


class AbsolutePositionEmbedding(nn.Module):
    """e_i = MLP(x,y,z,logA)：直接告诉每个 token 自己位于哪里（物理坐标 + 网格面积）。

    与 FaultPositionBias 的区别：这是加在特征上的绝对分量，不参与 pair 计算，
    因此代价 O(N) 而不是 O(N^2)。末层零初始化 → 训练起点与无该项时完全一致。
    """

    def __init__(self, dim, scatter=1.0, fourier=False, dropout=0.1, hidden=None):
        super().__init__()
        self.fourier = bool(fourier)
        self.freqs = (1.0, 0.3, 0.1) if self.fourier else ()
        in_dim = 4 + (6 * len(self.freqs) if self.fourier else 0)
        h = int(hidden or max(32, dim // 2))
        self.mlp = nn.Sequential(nn.Linear(in_dim, h), nn.SiLU(), nn.Linear(h, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.drop = nn.Dropout(float(dropout))
        self.scatter = float(scatter)

    def forward(self, xyz, area):
        n = xyz.shape[0]
        loga = torch.log(area.clamp_min(1e-8))
        loga = (loga - loga.mean()) / (loga.std() + 1e-6)
        feats = [xyz, loga[:, None]]
        if self.fourier:
            parts = [torch.sin(xyz * f * 3.141592653589793) for f in self.freqs]
            parts += [torch.cos(xyz * f * 3.141592653589793) for f in self.freqs]
            feats += parts
        e = self.mlp(torch.cat(feats, dim=-1))
        return self.drop(e) * self.scatter


class FaultPositionBias(nn.Module):
    """Continuous, directional absolute+relative pair bias, shared across a batch.

    Inputs are normalized physical coordinates. Final projection has no bias
    (a head-wise constant would cancel in softmax) and starts at zero.
    """

    def __init__(self, heads, hidden=16, limit=4.0, mode="mid_delta"):
        super().__init__()
        self.mode = str(mode)
        in_dim = 6                      # mid_delta: 3+3 ; relative_z: 3+3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, heads, bias=False)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        self.limit = float(limit)

    def forward(self, query_xyz, key_xyz, query_loga=None, key_loga=None):
        delta = key_xyz[None, :, :] - query_xyz[:, None, :]
        if self.mode == "relative_z":
            # 相对位移 + 范数 + 中点深度 + 面积比（不含 x/y 的绝对项，避免退化成"单元 ID"）
            norm = delta.norm(dim=-1, keepdim=True)
            z_mid = (query_xyz[:, 2][:, None] + key_xyz[None, :, 2]) * 0.5
            z_mid = z_mid.unsqueeze(-1)
            if query_loga is None or key_loga is None:
                log_ratio = torch.zeros_like(z_mid)
            else:
                log_ratio = (key_loga[None, :] - query_loga[:, None]).unsqueeze(-1)
            feats = torch.cat((delta, norm, z_mid, log_ratio), -1)
        else:
            mid = (query_xyz[:, None, :] + key_xyz[None, :, :]) * 0.5
            feats = torch.cat((mid, delta), -1)
        bias = self.mlp(feats)
        if self.limit > 0:
            bias = self.limit * torch.tanh(bias / self.limit)
        return bias.permute(2, 0, 1).unsqueeze(0)


class SpatialAttention(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv = int(config.num_key_value_heads)
        if dim % self.num_heads or self.num_heads % self.num_kv:
            raise ValueError(
                "Dimensions and grouped-query head counts must divide evenly"
            )
        self.head_dim = dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = float(config.dropout)
        self.chunk_size = int(getattr(config, "attention_chunk_size", 128))
        if self.chunk_size < 1:
            raise ValueError("attention_chunk_size must be positive")
        self.checkpoint_attention = bool(getattr(config, "checkpoint_attention", True))
        self.rotary_emb = (
            RotaryEmbedding(self.head_dim, int(config.max_seq_len))
            if getattr(config, "use_rope", True)
            else None
        )
        self.position_bias = (
            FaultPositionBias(
                self.num_heads,
                int(getattr(config, "position_bias_hidden", 16)),
                float(getattr(config, "position_bias_limit", 4.0)),
                str(getattr(config, "relative_bias_mode", "mid_delta")),
            )
            if getattr(config, "use_fault_position_bias", True)
            else None
        )
        # >0 时把注意力限制到每个 token 的 k 个空间近邻（gather 实现，非稠密 mask）
        self.local_k = int(getattr(config, "attention_local_k", 0) or 0)
        self.local_chunk = int(getattr(config, "attention_local_chunk", 256) or 256)
        self._knn_cache = {}
        self.residual_dropout = nn.Dropout(self.dropout)

    def _bias(self, q_xyz, k_xyz, q_loga=None, k_loga=None):
        if self.position_bias is None:
            return None
        return self.position_bias(q_xyz, k_xyz, q_loga, k_loga)

    def _bias_local(self, q_xyz, k_xyz, q_loga, k_loga):
        """局部注意力用的偏置：q_xyz (N,3), k_xyz (N,k,3) → (1,H,N,k)。

        逐 query 的邻居集合不同，无法用全局 (Q,K) 接口，这里直接复用偏置 MLP。
        """
        pb = self.position_bias
        if pb is None:
            return None
        delta = k_xyz - q_xyz[:, None, :]
        if pb.mode == "relative_z":
            norm = delta.norm(dim=-1, keepdim=True)
            z_mid = ((q_xyz[:, 2][:, None] + k_xyz[..., 2]) * 0.5).unsqueeze(-1)
            if q_loga is None or k_loga is None:
                log_ratio = torch.zeros_like(z_mid)
            else:
                log_ratio = (k_loga - q_loga[:, None]).unsqueeze(-1)
            feats = torch.cat((delta, norm, z_mid, log_ratio), -1)
        else:
            mid = (q_xyz[:, None, :] + k_xyz) * 0.5
            feats = torch.cat((mid, delta), -1)
        bias = pb.mlp(feats)
        if pb.limit > 0:
            bias = pb.limit * torch.tanh(bias / pb.limit)
        return bias.permute(2, 0, 1).unsqueeze(0)

    def _knn_index(self, query_xyz, key_xyz):
        key = (query_xyz.data_ptr(), key_xyz.data_ptr(), self.local_k, query_xyz.shape[0])
        cached = self._knn_cache.get(key)
        if cached is None:
            if cached is None and len(self._knn_cache) > 8:
                self._knn_cache.clear()
            dist = torch.cdist(query_xyz.detach(), key_xyz.detach())
            kk = min(self.local_k, int(query_xyz.shape[0]), int(key_xyz.shape[0]))
            cached = dist.topk(kk, largest=False).indices
            self._knn_cache[key] = cached
        return cached

    def _attend(self, q, k, v, query_xyz, key_xyz):
        bias = (
            self.position_bias(query_xyz, key_xyz).to(q.dtype)
            if self.position_bias is not None
            else None
        )
        # Query chunking bounds both attention and pair-MLP temporary arrays.
        # Backends are selected by PyTorch; dense bias need not support FlashAttention.
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    def forward(self, x, xyz, loga=None):
        b, n, dim = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, n, self.num_kv, self.head_dim)
        v = self.v_proj(x).reshape(b, n, self.num_kv, self.head_dim)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)
        q = q.transpose(1, 2)
        k = k.repeat_interleave(self.num_heads // self.num_kv, dim=2).transpose(1, 2)
        v = v.repeat_interleave(self.num_heads // self.num_kv, dim=2).transpose(1, 2)
        if self.local_k > 0:
            # 每个 query 只对自己最近的 k 个 key 做注意力：O(N*k) 而不是 O(N*N)。
            # 为了控制显存，query 侧分块（local_chunk），因为 gather 出来的
            # (B,H,c,k,d) 与 batch 成正比。
            idx_all = self._knn_index(xyz, xyz)                    # (N,k)
            kx_all = xyz[idx_all]                                  # (N,k,3)
            lg_all = None if loga is None else loga[idx_all]
            pieces = []
            step = self.local_chunk
            for st in range(0, n, step):
                en = min(st + step, n)
                ii = idx_all[st:en]                                # (c,k)
                kk = k[:, :, ii, :]                                # (B,H,c,k,d)
                vv = v[:, :, ii, :]
                bias = self._bias_local(xyz[st:en], kx_all[st:en],
                                        None if loga is None else loga[st:en],
                                        None if lg_all is None else lg_all[st:en])
                qq = q[:, :, st:en].unsqueeze(3)                   # (B,H,c,1,d)
                logits = (qq @ kk.transpose(-1, -2)).squeeze(3) / math.sqrt(self.head_dim)
                if bias is not None:
                    logits = logits + bias.to(q.dtype)
                attn = torch.softmax(logits.float(), dim=-1).to(q.dtype)
                if self.dropout > 0 and self.training:
                    attn = F.dropout(attn, p=self.dropout)
                pieces.append((attn.unsqueeze(3) @ vv).squeeze(3))  # (B,H,c,d)
            out = torch.cat(pieces, dim=2).transpose(1, 2).reshape(b, n, dim)
            return self.residual_dropout(self.o_proj(out))
        output = []
        for start in range(0, n, self.chunk_size):
            qc = q[:, :, start : start + self.chunk_size]
            coordinates = xyz[start : start + self.chunk_size]
            if self.training and torch.is_grad_enabled() and self.checkpoint_attention:
                value = checkpoint(
                    self._attend, qc, k, v, coordinates, xyz, use_reentrant=False
                )
            else:
                value = self._attend(qc, k, v, coordinates, xyz)
            output.append(value)
        out = torch.cat(output, dim=2).transpose(1, 2).reshape(b, n, dim)
        return self.residual_dropout(self.o_proj(out))


class GatedMLP(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        hidden = int(config.intermediate_size)
        self.gate_proj = nn.Linear(dim, hidden, bias=config.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden, bias=config.mlp_bias)
        self.down_proj = nn.Linear(hidden, dim, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SharedExpert(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, config.intermediate_size, bias=config.mlp_bias)
        self.w2 = nn.Linear(config.intermediate_size, dim, bias=config.mlp_bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class Gating(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.gate = nn.Linear(dim, config.expert_num)

    def forward(self, x):
        return self.gate(x)


class MoE(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.topk = int(config.topk)
        if not 1 <= self.topk <= int(config.expert_num):
            raise ValueError("Invalid MoE topk")
        self.experts = nn.ModuleList(
            [GatedMLP(config, dim) for _ in range(config.expert_num)]
        )
        self.shared_expert = SharedExpert(config, dim)
        self.gating = Gating(config, dim)

    def forward(self, x):
        logits = self.gating(x).reshape(-1, len(self.experts))
        selected, index = logits.topk(self.topk, -1)
        scores = F.softmax(selected.float(), -1).to(x.dtype)
        flat = x.reshape(-1, x.shape[-1])
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(index == expert_id)
            if rows.numel():
                contribution = expert(flat[rows]) * scores[rows, slots, None]
                out.index_add_(0, rows, contribution.to(out.dtype))
        return out.reshape_as(x) + self.shared_expert(x), logits


class SpatialLayer(nn.Module):
    def __init__(self, config, dim, index):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SpatialAttention(config, dim)
        self.norm2 = RMSNorm(dim)
        self.drop_path = DropPath(config.dropout)
        self.is_moe = bool(index % 2)
        if self.is_moe:
            self.moe = MoE(config, dim)
        else:
            self.mlp = GatedMLP(config, dim)

    def forward(self, x, xyz, loga=None):
        x = x + self.drop_path(self.self_attn(self.norm1(x), xyz, loga))
        if self.is_moe:
            out, logits = self.moe(self.norm2(x))
        else:
            out = self.mlp(self.norm2(x))
            logits = None
        return x + self.drop_path(out), logits


class SpatialBlock(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [SpatialLayer(config, dim, 0), SpatialLayer(config, dim, 1)]
        )
        self.checkpoint_blocks = bool(getattr(config, "checkpoint_blocks", False))

    def forward(self, x, xyz, loga=None):
        logits = None
        for layer in self.layers:
            if self.training and torch.is_grad_enabled() and self.checkpoint_blocks:
                x, logits = checkpoint(layer, x, xyz, loga, use_reentrant=False)
            else:
                x, logits = layer(x, xyz, loga)
        return x, logits


class ElementEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv1d(
            config.time_chans * config.data_chans, config.hidden_size, 1
        )

    def forward(self, x):
        b, t, c, n = x.shape
        return self.proj(x.reshape(b, t * c, n)).transpose(1, 2)


class SpatialDownSample(nn.Module):
    def __init__(self, dim, pairs=None):
        super().__init__()
        self.register_buffer(
            "pairs", None if pairs is None else torch.as_tensor(pairs, dtype=torch.long), persistent=False
        )
        self.norm = nn.LayerNorm(2 * dim)
        self.linear = nn.Linear(2 * dim, 2 * dim, bias=False)

    def forward(self, x, pairs=None):
        pairs = self.pairs if pairs is None else pairs
        if pairs is None:
            raise ValueError("Downsampling requires a geometry mapping")
        if x.ndim != 3:
            raise ValueError(f"Downsampling expects [B,N,D], got {tuple(x.shape)}")
        if pairs.device != x.device:
            raise ValueError("Downsampling mapping and features must be on the same device")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("Downsampling pairs must have shape [coarse_tokens,2]")
        if pairs.shape[0] < 1 or torch.any(pairs[:, 0] < 0):
            raise ValueError("Every coarse token must have a real first child")
        valid = pairs >= 0
        valid_indices = pairs[valid]
        if valid_indices.numel() and torch.any(valid_indices >= x.shape[1]):
            raise ValueError("Downsampling child index exceeds the fine-grid length")
        gathered = x[:, pairs.clamp_min(0), :]
        # A singleton is represented by its own feature in both slots.  This
        # keeps LayerNorm from treating a synthetic zero vector as a child.
        gathered = torch.where(
            valid[None, :, :, None], gathered, gathered[:, :, 0:1, :]
        )
        return self.linear(self.norm(gathered.flatten(2)))


class SpatialUpSample(nn.Module):
    def __init__(self, dim, parent=None, slot=None):
        super().__init__()
        self.register_buffer(
            "parent", None if parent is None else torch.as_tensor(parent, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "slot", None if slot is None else torch.as_tensor(slot, dtype=torch.long), persistent=False
        )
        self.norm = nn.LayerNorm(2 * dim)
        self.linear = nn.Linear(2 * dim, 2 * dim, bias=False)
        geom_hidden = max(16, dim // 4)
        # The coordinate tensors are already scaled for attention.  The
        # upsampling geometry uses a separate scale so typical child offsets
        # are O(1), while area ratios are centred around an equal-area pair.
        self.register_buffer(
            "relative_xyz_scale",
            torch.tensor([20.0, 20.0, 10.0], dtype=torch.float32),
            persistent=False,
        )
        self.area_log_center = math.log(0.5)
        self.area_log_scale = 0.5
        # Child identity is already represented by relative position and by
        # the parent/slot gather below.  Do not feed the arbitrary 0/1 slot
        # label into the geometric MLP.
        self.geometry_mlp = nn.Sequential(
            nn.Linear(4, geom_hidden), nn.SiLU(), nn.Linear(geom_hidden, dim)
        )
        # Start as an exact identity relative to the existing upsampling path.
        # The geometry residual is learned after the first optimizer update.
        nn.init.zeros_(self.geometry_mlp[-1].weight)
        nn.init.zeros_(self.geometry_mlp[-1].bias)
        self.dim = dim

    def forward(
        self,
        x,
        parent=None,
        slot=None,
        fine_xyz=None,
        coarse_xyz=None,
        fine_area=None,
        coarse_area=None,
    ):
        parent = self.parent if parent is None else parent
        slot = self.slot if slot is None else slot
        if parent is None or slot is None:
            raise ValueError("Upsampling requires a geometry mapping")
        if x.ndim != 3:
            raise ValueError(f"Upsampling expects [B,M,2D], got {tuple(x.shape)}")
        if parent.device != x.device or slot.device != x.device:
            raise ValueError("Upsampling mapping and features must be on the same device")
        if parent.ndim != 1 or slot.ndim != 1 or parent.shape != slot.shape:
            raise ValueError("Upsampling parent and slot must be matching [fine_cells] vectors")
        if parent.numel() < 1 or torch.any(parent < 0) or torch.any(parent >= x.shape[1]):
            raise ValueError("Upsampling parent index does not match the coarse-grid length")
        if torch.any(slot < 0) or torch.any(slot >= 2):
            raise ValueError("Upsampling child slots must be 0 or 1")
        has_geometry = all(
            value is not None
            for value in (fine_xyz, coarse_xyz, fine_area, coarse_area)
        )
        if not has_geometry and any(
            value is not None
            for value in (fine_xyz, coarse_xyz, fine_area, coarse_area)
        ):
            raise ValueError("Upsampling geometry arguments must be supplied together")
        children = self.linear(self.norm(x)).reshape(
            x.shape[0], x.shape[1], 2, self.dim
        )
        children = children[:, parent, slot, :]
        if not has_geometry:
            # Keep the small standalone mapping utility backward compatible;
            # the model itself always supplies geometry below.
            return children
        for name, value in (("fine_xyz", fine_xyz), ("coarse_xyz", coarse_xyz),
                            ("fine_area", fine_area), ("coarse_area", coarse_area)):
            if value.device != x.device:
                raise ValueError(f"{name} and features must be on the same device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if fine_xyz.ndim != 2 or fine_xyz.shape != (parent.numel(), 3):
            raise ValueError("fine_xyz must have shape [fine_cells,3] matching parent")
        if coarse_xyz.ndim != 2 or coarse_xyz.shape != (x.shape[1], 3):
            raise ValueError("coarse_xyz must have shape [coarse_tokens,3] matching x")
        if fine_area.ndim != 1 or fine_area.shape != (parent.numel(),) or torch.any(fine_area <= 0):
            raise ValueError("fine_area must be positive with one value per fine cell")
        if coarse_area.ndim != 1 or coarse_area.shape != (x.shape[1],) or torch.any(coarse_area <= 0):
            raise ValueError("coarse_area must be positive with one value per coarse token")
        relative_xyz = (fine_xyz - coarse_xyz[parent]) * self.relative_xyz_scale.to(
            fine_xyz.dtype
        )
        log_area_ratio = (
            torch.log(fine_area.clamp_min(1e-8) / coarse_area[parent].clamp_min(1e-8))
            - self.area_log_center
        ).div(self.area_log_scale).unsqueeze(-1)
        geometry = torch.cat((relative_xyz, log_area_ratio), dim=-1)
        geometry_bias = self.geometry_mlp(geometry).to(children.dtype)
        return children + geometry_bias[None, :, :]


class FaultSpatialModel(nn.Module):
    """Raw output [B, H*variables, elements]; original element order is preserved."""

    def __init__(self, config, geometry=None):
        super().__init__()
        if int(config.patch_size) != 1:
            raise ValueError("FaultSpatialModel requires patch_size=1")
        self.config = config
        self.dynamic_geometry = bool(getattr(config, "dynamic_geometry", False))
        if self.dynamic_geometry:
            if getattr(config, "use_learned_position", True) or getattr(config, "use_rope", True):
                raise ValueError("Dynamic geometry requires disabling ID position table and RoPE")
            geometry = None
        else:
            geometry = (load_geometry(config.params) if geometry is None else geometry).validate()
            if len(geometry.fine_xyz_km) != int(config.fea_len):
                raise ValueError("Geometry and input element counts disagree")
        self.geometry_metadata = {} if geometry is None else dict(geometry.metadata)
        scales = torch.as_tensor(
            getattr(config, "position_coordinate_scale_km", (100.0, 100.0, 20.0)),
            dtype=torch.float32,
        )
        if (
            scales.shape != (3,)
            or not torch.isfinite(scales).all()
            or torch.any(scales <= 0)
        ):
            raise ValueError(
                "position_coordinate_scale_km must contain three positive scales"
            )
        self.register_buffer(
            "fine_xyz",
            None if geometry is None else torch.tensor(geometry.fine_xyz_km, dtype=torch.float32) / scales,
            persistent=False,
        )
        self.register_buffer(
            "coarse_xyz",
            None if geometry is None else torch.tensor(geometry.coarse_xyz_km, dtype=torch.float32) / scales,
            persistent=False,
        )
        self.register_buffer(
            "fine_area",
            None if geometry is None else torch.tensor(geometry.fine_area, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "coarse_area",
            None if geometry is None else torch.tensor(geometry.coarse_area, dtype=torch.float32),
            persistent=False,
        )
        depth_weight = None
        if geometry is not None and getattr(config, "depth_weighting", False):
            # Single source of truth: the joint profile is what the loss uses
            # for every region.  A previous legacy table is intentionally not
            # reachable from here (cascadia is numerically identical; alaska
            # moves from 35 km to the joint 45 km peak).
            depth_weight = torch.tensor(
                joint_depth_weights(geometry.fine_xyz_km[:, 2], config.region)
            )
        self.register_buffer("depth_weights", depth_weight, persistent=False)
        dim = int(config.hidden_size)
        self.patch_embed = ElementEmbedding(config)
        self.pos_embed = (
            nn.Parameter(torch.empty(1, config.fea_len, dim))
            if getattr(config, "use_learned_position", True)
            else None
        )
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(config.dropout)
        self.pos_embed_abs = None
        self.pos_embed_abs_coarse = None
        if str(getattr(config, "position_embedding", "none")) == "absolute":
            self.pos_embed_abs = AbsolutePositionEmbedding(
                dim,
                float(getattr(config, "position_embedding_scale", 1.0)),
                bool(getattr(config, "position_embedding_fourier", False)),
                float(getattr(config, "position_embedding_dropout", 0.1)),
            )
            self.pos_embed_abs_coarse = AbsolutePositionEmbedding(
                2 * dim,
                float(getattr(config, "position_embedding_scale", 1.0)),
                bool(getattr(config, "position_embedding_fourier", False)),
                float(getattr(config, "position_embedding_dropout", 0.1)),
            )
        self.layer1 = SpatialBlock(config, dim)
        self.layer2 = SpatialBlock(config, 2 * dim)
        self.layer3 = SpatialBlock(config, 2 * dim)
        self.layer4 = SpatialBlock(config, dim)
        self.downsample = SpatialDownSample(dim, None if geometry is None else geometry.pairs)
        self.upsample = SpatialUpSample(dim, None if geometry is None else geometry.parent, None if geometry is None else geometry.slot)
        self.head1 = nn.Linear(dim, 256)
        self.head2 = nn.Linear(256, 64)
        self.head_out = nn.Linear(64, int(config.out_chans))
        self.act = nn.GELU()
        self.head_drop = nn.Dropout(config.dropout)
        self.signature = {
            "model_version": MODEL_VERSION,
            "geometry_hash": None if geometry is None else geometry.metadata["geometry_hash"],
            "fea_len": int(config.fea_len),
            "time_chans": int(config.time_chans),
            "data_chans": int(config.data_chans),
            "out_chans": int(config.out_chans),
            "hidden_size": dim,
            "heads": int(config.num_attention_heads),
            "kv_heads": int(config.num_key_value_heads),
            "position_coordinate_scale_km": scales.tolist(),
            "use_rope": bool(getattr(config, "use_rope", True)),
            "use_fault_position_bias": bool(
                getattr(config, "use_fault_position_bias", True)
            ),
            "use_learned_position": bool(getattr(config, "use_learned_position", True)),
            "intermediate_size": int(config.intermediate_size),
            "expert_num": int(config.expert_num),
            "topk": int(config.topk),
            "dropout": float(config.dropout),
            "position_bias_hidden": int(getattr(config, "position_bias_hidden", 16)),
            "position_bias_limit": float(getattr(config, "position_bias_limit", 4.0)),
            "geometry_conditioned_upsample": True,
            "geometry_upsample_relative_xyz_scale": [20.0, 20.0, 10.0],
            "geometry_upsample_area_log_center": math.log(0.5),
            "geometry_upsample_area_log_scale": 0.5,
            "geometry_upsample_uses_slot_feature": False,
            "depth_weighting": bool(getattr(config, "depth_weighting", False)),
            "region": str(getattr(config, "region", "cascadia")),
        }

        if self.dynamic_geometry:
            for key in ("geometry_hash", "fea_len", "region", "depth_weighting"):
                self.signature.pop(key)
            self.signature["model_version"] = MODEL_VERSION + "_joint"

    def get_extra_state(self):
        return dict(self.signature)

    def set_extra_state(self, state):
        if state != self.signature:
            raise ValueError(
                "Checkpoint geometry or model signature disagrees with this model. "
                "Use its saved configuration and exact fault ordering."
            )

    def _balance(self, all_logits):
        if not self.config.output_router_logits:
            return None
        logits = torch.cat(all_logits, 0)
        probs = F.softmax(logits.float(), -1)
        idx = probs.topk(self.config.topk, -1).indices
        mask = F.one_hot(idx, self.config.expert_num).float()
        return (mask.mean(0) * probs.mean(0)[None, :]).sum() * self.config.expert_num

    def forward(self, x, geometry=None):
        if self.dynamic_geometry and geometry is None:
            raise ValueError("Joint model requires external geometry")
        if not self.dynamic_geometry and geometry is not None:
            raise ValueError("Static model cannot switch geometry")
        n = len(geometry.fine_xyz) if self.dynamic_geometry else int(self.config.fea_len)
        fine = geometry.fine_xyz if self.dynamic_geometry else self.fine_xyz
        coarse = geometry.coarse_xyz if self.dynamic_geometry else self.coarse_xyz
        if fine.device != x.device or coarse.device != x.device:
            raise ValueError("Input and geometry must be on the same device")
        if self.dynamic_geometry and geometry.metadata.get("coordinate_scales") != self.signature["position_coordinate_scale_km"]:
            raise ValueError("Geometry coordinate scale mismatch")
        expected = (
            int(self.config.time_chans),
            int(self.config.data_chans),
            n,
        )
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected input [B,{expected}], got {tuple(x.shape)}")
        fine_area = geometry.fine_area if self.dynamic_geometry else self.fine_area
        coarse_area = geometry.coarse_area if self.dynamic_geometry else self.coarse_area
        need_loga = (self.pos_embed_abs is not None) or any(
            getattr(blk.layers[0].self_attn, "position_bias", None) is not None
            and blk.layers[0].self_attn.position_bias.mode == "relative_z"
            for blk in (self.layer1, self.layer4)
        )
        fine_loga = torch.log(fine_area.clamp_min(1e-8)) if need_loga else None
        coarse_loga = torch.log(coarse_area.clamp_min(1e-8)) if need_loga else None
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed.to(x.dtype)
        if self.pos_embed_abs is not None:
            x = x + self.pos_embed_abs(fine, fine_area).to(x.dtype)[None]
        x = self.pos_drop(x)
        x, g1 = self.layer1(x, fine, fine_loga)
        skip = x
        x = self.downsample(x, geometry.pairs if self.dynamic_geometry else None)
        if self.pos_embed_abs_coarse is not None:
            x = x + self.pos_embed_abs_coarse(coarse, coarse_area).to(x.dtype)[None]
        x, g2 = self.layer2(x, coarse, coarse_loga)
        x, g3 = self.layer3(x, coarse, coarse_loga)
        x = self.upsample(
            x,
            geometry.parent if self.dynamic_geometry else None,
            geometry.slot if self.dynamic_geometry else None,
            fine,
            coarse,
            fine_area,
            coarse_area,
        ) + skip
        x, g4 = self.layer4(x, fine, fine_loga)
        x = self.head_drop(self.act(self.head1(x)))
        x = self.head_drop(self.act(self.head2(x)))
        x = self.head_out(x).transpose(1, 2).contiguous()
        return x, self._balance([g1, g2, g3, g4])


class JointFaultSpatialModel(FaultSpatialModel):
    """One parameter set for variable meshes; predicts all daily states directly."""
    def forward(self, x, geometry):
        raw, aux = super().forward(x, geometry)
        raw = raw.reshape(x.shape[0], self.config.horizon, self.config.data_chans,
                          x.shape[-1])
        if bool(getattr(self.config, "residual_output", False)):
            # 残差参数化：模型输出的是"相对最后一天的变化量"，
            #   X_hat[t] = X_last + delta_hat[t]
            # 这样 (a) 不复制最后一天也能起步，(b) 让增量项成为直接可优化的量。
            # 注意 x 与 raw 都在 z-score 归一化空间，可直接相加。
            raw = raw + x[:, -1][:, None, :, :]
        return raw, aux
