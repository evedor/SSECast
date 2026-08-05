import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from transformers import PretrainedConfig
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
        self.rotary_emb = RotaryEmbedding(self.head_dim)
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

        # output = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
        #                                             dropout_p=self.dropout if self.training else 0.0,
        #                                             is_causal=self.is_causal)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
        scores = scores + self.b
        scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax
        scores = self.attention_dropout(scores)  # 应用注意力 dropout
        output = torch.matmul(scores, v)  # 计算输出

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

    def forward(self, x):
        B, N, C = x.shape  # x: (batch_size, in_length, in_dim)
        x = x.transpose(1, 2)  # Shape: (B, C, in_length)
        x = F.interpolate(x, size=self.out_length, mode='linear', align_corners=False)  # Shape: (B, C, out_length)
        x = x.transpose(1, 2)  # Shape: (B, out_length, C)
        # if self.pad is not None:
        #     x = self.pad(x)  # Shape: (B, out_length, C)
        x = self.norm(x)  # Shape: (B, out_length, C)
        x = self.linear(x)  # Shape: (B, out_length, C // 2)

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
        x = x.transpose(1, 2)
        x = self.pad(x)  # Shape: (B, out_length * 2, C)
        x = x.transpose(1, 2)
        x = x.view(B, self.out_length, 2, C)  # Shape: (B, out_length, 2, C)
        x = x.reshape(B, self.out_length, 2 * C)  # Shape: (B, out_length, 2 * C)
        x = self.norm(x)
        x = self.linear(x)  # Shape: (B, out_length, 2 * C)
        return x

class SSEPREModel(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.dropout)
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
        
        self.act = nn.GELU() 
        self.head_drop = nn.Dropout(config.dropout)


        # self.load_balancing_loss_func = load_balancing_loss_func()

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

    def forward(self, x):
        all_router_logits = [] if self.config.output_router_logits else None

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        
        x, router_logits_1 = self.layer1(x)
        skip = x

        x = self.downsample(x)
        x, router_logits_2 = self.layer2(x)
        x, router_logits_3 = self.layer3(x)
        x = self.upsample(x)
        x = x + skip
        x, router_logits_4 = self.layer4(x)
        # x, router_logits_5 = self.layer5(x)

        # x = torch.concat([x, skip], dim=-1)

        x = self.head1(x)
        x = self.act(x)      
        x = self.head_drop(x)
        
        x = self.head2(x)
        x = self.act(x)       
        x = self.head_drop(x) 
        
        x = self.head_out(x)
        x = rearrange(
            x,
            "b l (p c_out) -> b c_out (l p)",
            p=self.config.patch_size,
            l=self.config.fea_len // self.config.patch_size,
        )
        if self.config.output_router_logits:
            all_router_logits.extend([router_logits_1, router_logits_2, router_logits_3, router_logits_4]) #

        aux_loss = None
        if self.config.output_router_logits:
            aux_loss = self.load_balancing_loss_func(all_router_logits, self.config.expert_num, self.config.topk)

        return x, aux_loss


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = config.fea_len // config.patch_size
        self.fea_len = config.fea_len
        self.patch_size = config.patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv1d(config.time_chans * config.data_chans, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, D, C, L = x.shape #1,2,3,3339
        x = x.view(B, D * C, L)
        # assert L == self.fea_len, f"Input signal length ({L}) doesn't match model ({self.fea_len})."
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