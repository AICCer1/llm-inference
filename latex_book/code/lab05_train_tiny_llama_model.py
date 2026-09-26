"""
Lab 05: 一个 Llama 架构的迷你语言模型（每个组件都对应教程里的一个推导）

  RMSNorm      → 第 5 篇 §5.2（为什么去掉减均值）、第 8 篇 §6（为什么求和要转 FP32）
  RoPE         → 第 4 篇 §6（相对位置恒等式）、第 5 篇 §5.4（K 旋转后再进缓存）
  GQA          → 第 5 篇 §5.5、第 8 篇 §4.3（KV 头少于 Q 头）
  因果掩码     → 第 4 篇 §4
  SwiGLU       → 第 5 篇 §5.3
  Pre-LN 残差  → 第 3 篇 §7、第 5 篇 §5.1
  KV Cache     → 第 6 篇 §3

只用 torch 的基本张量运算和 nn.Linear / nn.Embedding，不用 nn.MultiheadAttention 等高层封装。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 8          # Q 头数
    n_kv_head: int = 2       # KV 头数（GQA：每 4 个 Q 头共享 1 组 KV）
    d_model: int = 256
    ffn_hidden: int = 688    # ≈ 8/3 * d_model，取 16 的倍数（第 5 篇 §5.3）
    max_seq_len: int = 1024  # RoPE 表预先算到这么长
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0     # 只在训练时生效（model.eval() 后自动关闭），推理代码完全不受影响

    @property
    def head_dim(self):
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 求均方要在 FP32 里做，BF16 累加会丢精度（第 8 篇 §6.3）
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        # 注意这里【不要】写成 xf.to(x.dtype) * self.weight：
        #   weight 是 FP32 参数，乘上去会把结果再提升回 FP32 —— 于是那次 .to() 只起到
        #   "把归一化后的值先截断到 BF16 再算"的作用，纯属有损往返，输出依然是 FP32。
        # 所以整个残差流是 FP32，autocast 只作用在子层内部的 Linear 上。这比纯 BF16 更准，
        # 代价是显存/带宽占用更高。想改成真正的 BF16 残差流，需要同时处理 nn.Embedding
        # （autocast 下它同样输出 FP32，残差加法会再把它提升回去），只改本行是不够的。
        return xf * self.weight


def precompute_rope(head_dim, max_len, base):
    """返回 cos/sin 表，形状 [max_len, head_dim/2]，第 m 行第 i 列是 cos(m * base^(-2i/d))"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """
    x: [B, H, T, D]；cos/sin: [T, D/2]
    采用 Hugging Face / Llama 的"前后两半配对"写法：第 i 维与第 i + D/2 维组成一对做 2D 旋转。
    （lab04_transformer_microscope 里用的是相邻两维配对，两者只是维度排列不同，数学性质完全一样）
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None, start_pos=0):
        """
        x: [B, T, d_model]，T 在 Prefill / 训练时是整段长度，在 Decode 时是 1
        cache: None（训练/朴素推理）或一个 dict {"k": [B, n_kv, S_past, hd], "v": ...}，原地更新
        start_pos: 本次输入的第一个 token 在整个序列中的位置（决定 RoPE 转多少度、掩码怎么画）
        """
        B, T, _ = x.shape
        cfg, hd = self.cfg, self.cfg.head_dim
        q = self.wq(x).view(B, T, cfg.n_head, hd).transpose(1, 2)      # [B, n_head, T, hd]
        k = self.wk(x).view(B, T, cfg.n_kv_head, hd).transpose(1, 2)   # [B, n_kv,   T, hd]
        v = self.wv(x).view(B, T, cfg.n_kv_head, hd).transpose(1, 2)

        # RoPE：只转 Q 和 K，按它们各自的绝对位置转；K 转好之后才写进缓存，以后再也不用动它
        pos_cos, pos_sin = cos[start_pos:start_pos + T], sin[start_pos:start_pos + T]
        q = apply_rope(q, pos_cos, pos_sin)
        k = apply_rope(k, pos_cos, pos_sin)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        S = k.shape[2]  # 总共能看到的 key 数 = start_pos + T

        # GQA：把每个 KV 头复制给它负责的 n_head / n_kv_head 个 Q 头
        rep = cfg.n_head // cfg.n_kv_head
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)             # [B, n_head, T, S]
        # 因果掩码：第 i 个 query 的绝对位置是 start_pos + i，只能看绝对位置 <= 它的 key
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(1)
        k_pos = torch.arange(S, device=x.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)           # Softmax 在 FP32 里做
        out = probs @ v                                                 # [B, n_head, T, hd]
        out = out.transpose(1, 2).reshape(B, T, cfg.n_head * hd)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, start_pos=0):
        # Pre-LN：残差这条"高速公路"上什么都不放
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, cache, start_pos))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


class TinyLlama(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # 小标准差初始化：让初始 logits 接近 0 → 初始分布接近均匀 → 初始损失 ≈ ln(V)（第 3 篇 §5.3）
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def new_cache(self):
        """每层一个空 dict；Attention.forward 会往里面写入 k/v"""
        return [dict() for _ in self.blocks]

    def forward(self, idx, cache=None, start_pos=0):
        """idx: [B, T] 的 token id；返回 logits [B, T, vocab]"""
        x = self.tok_emb(idx)
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.rope_cos, self.rope_sin, None if cache is None else cache[i], start_pos)
        return self.lm_head(self.norm(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
