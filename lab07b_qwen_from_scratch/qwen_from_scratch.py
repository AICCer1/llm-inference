"""
Lab 07b: 不用 transformers 的模型类，只拿原始权重张量，亲手实现 Qwen2.5 的完整推理

这是整条学习路线里"从使用者变成理解者"的关键一步：
  - transformers 只用来做两件事：(1) 分词器 (2) 作为"标准答案"和我们对拍
  - 前向传播、RoPE、GQA、KV Cache、采样全部自己写，每一行都能对应到教程的公式

实验清单:
  1. 读 config.json，逐字段解释（第 5 篇的自测题）
  2. 读 safetensors，看清每一层有哪些权重、形状是什么、参数都花在哪了
  3. 看 BPE 分词器怎么切中文（第 5 篇 §2）
  4. 手写前向传播，与 HF 官方实现逐位对拍 logits（FP32 下误差应 < 1e-3）
  5. 手写 KV Cache 贪婪生成，与 HF generate() 逐 token 对拍
  6. 性能：Prefill 吞吐、Decode 每 token 耗时，与第 8 篇的带宽下界对比；朴素 vs KV Cache

运行:
  .venv/bin/python lab07b_qwen_from_scratch/qwen_from_scratch.py
第一次运行会从 Hugging Face 下载 Qwen/Qwen2.5-0.5B-Instruct（约 1 GB）。
国内网络可先设置镜像: export HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def title(s):
    print("\n" + "=" * 80)
    print(f"🔬 {s}")
    print("=" * 80)


# =====================================================================
# 实验 1 & 2: 配置与权重
# =====================================================================
CONFIG_NOTES = {
    "hidden_size": "d_model，残差流的宽度",
    "num_hidden_layers": "Transformer 层数 L",
    "num_attention_heads": "Q 头数 n_h",
    "num_key_value_heads": "KV 头数 n_kv（< n_h 说明是 GQA，第 5 篇 §5.5）",
    "intermediate_size": "SwiGLU 中间维度（第 5 篇 §5.3）",
    "vocab_size": "词表大小 V（第 5 篇 §2）",
    "rope_theta": "RoPE 底数 b，越大越适合长上下文（第 5 篇 §5.4）",
    "rms_norm_eps": "RMSNorm 分母里的 ε",
    "tie_word_embeddings": "词嵌入与 LM Head 共享同一个矩阵（第 5 篇 §5.7）",
    "max_position_embeddings": "设计支持的最长上下文",
    "torch_dtype": "权重存储精度（第 8 篇 §6）",
    "hidden_act": "SwiGLU 门控用的激活函数",
}


def exp1_config(model_dir):
    title("实验 1: 逐字段读懂 config.json")
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    for k, note in CONFIG_NOTES.items():
        print(f"   {k:<26} = {str(cfg[k]):<12} ← {note}")
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    kv_per_tok = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2
    print(f"\n   推导量: head_dim = {cfg['hidden_size']} / {cfg['num_attention_heads']} = {hd}；"
          f"GQA 分组 = {cfg['num_attention_heads'] // cfg['num_key_value_heads']}")
    print(f"   BF16 每 token KV Cache = 2 × {cfg['num_hidden_layers']} × {cfg['num_key_value_heads']} × {hd} × 2B = {kv_per_tok:,} 字节 ≈ {kv_per_tok / 1024:.0f} KB")
    print(f"   → 在 16GB 显卡上，扣掉 1GB 权重后，理论上能缓存约 {14 * 1024**3 // kv_per_tok / 1e3:.0f}K 个 token")
    return cfg


def exp2_weights(model_dir, cfg):
    title("实验 2: 打开 safetensors，看看参数都花在哪了")
    W = load_file(os.path.join(model_dir, "model.safetensors"))
    total = sum(t.numel() for t in W.values())
    print(f"   共 {len(W)} 个张量，{total / 1e6:.1f}M 参数，存储类型 {next(iter(W.values())).dtype}")
    print("\n   第 0 层的全部权重（注意 nn.Linear 的权重形状是 [out, in]）:")
    for name, t in W.items():
        if name.startswith("model.layers.0."):
            print(f"      {name[len('model.layers.0.'):]:<34} {str(list(t.shape)):<14} {t.numel() / 1e6:6.2f}M")
    emb = W["model.embed_tokens.weight"].numel()
    L = cfg["num_hidden_layers"]
    attn = sum(t.numel() for n, t in W.items() if ".self_attn." in n)
    mlp = sum(t.numel() for n, t in W.items() if ".mlp." in n)
    print(f"\n   参数分布: 词嵌入 {emb / 1e6:.1f}M ({emb / total:.0%}) | 注意力 {attn / 1e6:.1f}M ({attn / total:.0%}) | "
          f"FFN {mlp / 1e6:.1f}M ({mlp / total:.0%}) | 其余 {(total - emb - attn - mlp) / 1e6:.2f}M")
    print(f"   👉 小模型里词表矩阵占了 {emb / total:.0%}！这就是为什么 0.5B 模型要让 LM Head 与词嵌入共享权重。")
    print(f"      注意力只占 {attn / total:.0%}：GQA 让 K/V 投影矩阵很小（{cfg['num_key_value_heads']} 个头 vs Q 的 {cfg['num_attention_heads']} 个）。")
    if "lm_head.weight" not in W:
        print("   （文件里没有 lm_head.weight —— 因为 tie_word_embeddings=true，推理时直接复用 embed_tokens）")
    return W


# =====================================================================
# 手写 Qwen2 的前向传播（本文件的核心，约 80 行）
# =====================================================================
class QwenFromScratch:
    def __init__(self, W, cfg, device, dtype):
        self.cfg = cfg
        self.L = cfg["num_hidden_layers"]
        self.n_h = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["hidden_size"] // self.n_h
        self.eps = cfg["rms_norm_eps"]
        self.dtype, self.device = dtype, device
        self.W = {k: v.to(device=device, dtype=dtype) for k, v in W.items()}
        if "lm_head.weight" not in self.W:
            self.W["lm_head.weight"] = self.W["model.embed_tokens.weight"]   # 权重共享
        # RoPE 频率表：theta_i = base^(-2i/d)，算到 32K 位置
        inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))
        ang = torch.outer(torch.arange(32768, device=device).float(), inv_freq)   # [pos, hd/2]
        self.cos, self.sin = ang.cos(), ang.sin()

    def rms_norm(self, x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(self.dtype) * w

    def rope(self, x, pos0):
        """x: [B, H, T, hd]；前一半与后一半配对旋转（与 HF 的 rotate_half 等价）"""
        T = x.shape[2]
        cos = self.cos[pos0:pos0 + T].to(x.dtype)
        sin = self.sin[pos0:pos0 + T].to(x.dtype)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def new_cache(self, max_len):
        """预分配 KV Cache：每层一块 [1, n_kv, max_len, hd]，按位置写入（比 torch.cat 反复拼接更接近真实引擎）"""
        shape = (1, self.n_kv, max_len, self.hd)
        return [(torch.empty(shape, device=self.device, dtype=self.dtype),
                 torch.empty(shape, device=self.device, dtype=self.dtype)) for _ in range(self.L)]

    @torch.inference_mode()
    def forward(self, ids, cache=None, pos0=0):
        """ids: [1, T]；cache 为 None 时做不带缓存的完整前向；返回 logits [1, T, V]"""
        W, P = self.W, "model.layers."
        B, T = ids.shape
        x = F.embedding(ids, W["model.embed_tokens.weight"])                        # [B, T, d]
        for i in range(self.L):
            p = f"{P}{i}."
            # ---------------- 注意力子层 ----------------
            h = self.rms_norm(x, W[p + "input_layernorm.weight"])
            q = F.linear(h, W[p + "self_attn.q_proj.weight"], W[p + "self_attn.q_proj.bias"])   # Qwen2 的 QKV 带 bias
            k = F.linear(h, W[p + "self_attn.k_proj.weight"], W[p + "self_attn.k_proj.bias"])
            v = F.linear(h, W[p + "self_attn.v_proj.weight"], W[p + "self_attn.v_proj.bias"])
            q = q.view(B, T, self.n_h, self.hd).transpose(1, 2)                        # [B, n_h,  T, hd]
            k = k.view(B, T, self.n_kv, self.hd).transpose(1, 2)                       # [B, n_kv, T, hd]
            v = v.view(B, T, self.n_kv, self.hd).transpose(1, 2)
            q, k = self.rope(q, pos0), self.rope(k, pos0)                              # K 旋转后再进缓存

            if cache is not None:
                kc, vc = cache[i]
                kc[:, :, pos0:pos0 + T] = k
                vc[:, :, pos0:pos0 + T] = v
                k, v = kc[:, :, :pos0 + T], vc[:, :, :pos0 + T]
            S = k.shape[2]

            rep = self.n_h // self.n_kv                                                 # GQA 展开
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)                    # [B, n_h, T, S]
            if T > 1:   # Decode（T=1）时新 token 能看到全部历史，不需要掩码
                q_pos = torch.arange(pos0, pos0 + T, device=ids.device)[:, None]
                k_pos = torch.arange(S, device=ids.device)[None, :]
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
            attn = F.softmax(scores.float(), dim=-1).to(self.dtype) @ v                # [B, n_h, T, hd]
            attn = attn.transpose(1, 2).reshape(B, T, self.n_h * self.hd)
            x = x + F.linear(attn, W[p + "self_attn.o_proj.weight"])                   # 残差 1
            # ---------------- SwiGLU 子层 ----------------
            h = self.rms_norm(x, W[p + "post_attention_layernorm.weight"])
            gate = F.linear(h, W[p + "mlp.gate_proj.weight"])
            up = F.linear(h, W[p + "mlp.up_proj.weight"])
            x = x + F.linear(F.silu(gate) * up, W[p + "mlp.down_proj.weight"])         # 残差 2
        x = self.rms_norm(x, W["model.norm.weight"])
        return F.linear(x, W["lm_head.weight"])                                        # [B, T, V]

    @torch.inference_mode()
    def generate(self, ids, max_new, eos_ids=(), use_cache=True, repetition_penalty=1.0):
        """贪婪解码。返回 (新 token 列表, prefill 秒数, 每步 decode 秒数列表)
        repetition_penalty: 第 7 篇 §5 的公式，对【提示词 + 已生成】里出现过的 token：正 logit 除以 θ，负 logit 乘以 θ"""
        ids = ids.to(self.device)
        out, dec_times = [], []
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        cache = self.new_cache(ids.shape[1] + max_new) if use_cache else None

        sync(); t0 = time.perf_counter()
        logits = self.forward(ids, cache, 0)[:, -1]                                   # Prefill
        sync(); t_prefill = time.perf_counter() - t0
        seq = ids
        seen = ids[0].tolist()
        for step in range(max_new):
            if repetition_penalty != 1.0:
                idx = torch.tensor(sorted(set(seen)), device=self.device)
                sc = logits[0, idx]
                logits[0, idx] = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
            nxt = logits.argmax(-1, keepdim=True)                                      # [1, 1]
            seen.append(int(nxt))
            out.append(int(nxt))
            if int(nxt) in eos_ids:
                break
            sync(); t0 = time.perf_counter()
            if use_cache:
                logits = self.forward(nxt, cache, ids.shape[1] + step)[:, -1]         # 只送 1 个 token
            else:
                seq = torch.cat([seq, nxt], dim=1)
                logits = self.forward(seq)[:, -1]                                      # 整段重算
            sync(); dec_times.append(time.perf_counter() - t0)
        return out, t_prefill, dec_times


# =====================================================================
# 实验 3 ~ 6
# =====================================================================
def exp3_tokenizer(tok):
    title("实验 3: BPE 分词器是怎么切中文的（第 5 篇 §2）")
    for text in ["大模型推理的瓶颈是显存带宽。", "KV Cache", "Transformer", "饕餮"]:
        ids = tok.encode(text)
        pieces = [tok.decode([i]) for i in ids]
        print(f"   {text!r:<22} → {len(ids):2d} 个 token: {pieces}")
    print("   👉 常见词整体成一个 token，罕见字被拆成多个字节级片段（解码单个片段可能显示为乱码 �）。")


def exp4_logits_match(model_dir, tok, W, cfg, device):
    title("实验 4: 手写前向 vs HF 官方实现 —— 逐位对拍 logits (FP32)")
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    mine = QwenFromScratch(W, cfg, device, torch.float32)
    ids = torch.tensor([tok.encode("大模型推理的本质是自回归地预测下一个 token，")], device=device)
    with torch.inference_mode():
        ref_logits = ref(ids).logits
    my_logits = mine.forward(ids)
    diff = (ref_logits - my_logits).abs().max().item()
    same_top1 = bool((ref_logits.argmax(-1) == my_logits.argmax(-1)).all())
    print(f"   输入 {ids.shape[1]} 个 token，logits 形状 {list(my_logits.shape)}")
    print(f"   最大绝对误差 = {diff:.2e}，每个位置的 top-1 预测{'全部一致 ✅' if same_top1 else '存在不一致 ❌'}")
    assert diff < 1e-3 and same_top1, '手写前向与 HF 官方不一致'
    print("   👉 误差只来自浮点运算顺序不同。你写的每一行（RMSNorm、RoPE、GQA、bias、SwiGLU、权重共享）都和官方一致。")
    return ref, mine


def exp5_generate_match(tok, ref, mine, device):
    title("实验 5: 手写 KV Cache 贪婪生成 vs HF generate() —— 逐 token 对拍")
    msgs = [{"role": "user", "content": "用两句话解释什么是 KV Cache。"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok.encode(text)], device=device)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    print(f"   提示词（套用聊天模板后 {ids.shape[1]} 个 token）: {msgs[0]['content']}")
    print(f"   聊天模板展开后的真实输入: {text!r}")

    def compare(tag, ref_kwargs, my_penalty):
        with torch.inference_mode():
            ref_out = ref.generate(ids, max_new_tokens=60, do_sample=False, **ref_kwargs)[0, ids.shape[1]:].tolist()
        my_out, _, _ = mine.generate(ids, 60, eos_ids=eos, repetition_penalty=my_penalty)
        ref_out = [t for t in ref_out if t != tok.pad_token_id]
        print(f"\n   【{tag}】")
        print(f"   我的实现: {tok.decode(my_out, skip_special_tokens=True)}")
        print(f"   HF 官方 : {tok.decode(ref_out, skip_special_tokens=True)}")
        print(f"   {len(my_out)} vs {len(ref_out)} 个 token，{'逐 token 完全一致 ✅' if my_out == ref_out else '不一致 ❌'}")
        assert my_out == ref_out, f'{tag}: 生成结果与 HF 官方不一致'

    compare("纯贪婪: 显式关闭重复惩罚", dict(repetition_penalty=1.0), 1.0)
    compare("HF 默认行为: generation_config.json 里藏着 repetition_penalty=1.1", dict(), 1.1)
    print("\n   👉 两个教训:")
    print("      1. 只写 do_sample=False 时，HF 仍然会读取模型自带的 generation_config.json，偷偷套用 repetition_penalty=1.1。")
    print("         '用库'时这类隐藏默认值随处可见；自己实现一遍才知道每个输出到底是怎么来的。")
    print("      2. 我们按第 7 篇 §5 的公式手写重复惩罚后，与官方再次逐 token 一致 —— 公式就是全部真相。")
    return ids


def exp6_perf(W, cfg, tok, device, bw_gbs):
    title("实验 6: 性能 —— Prefill 吞吐、Decode 耗时与第 8 篇的带宽下界 (BF16)")
    mine = QwenFromScratch(W, cfg, device, torch.bfloat16)
    n_params = sum(t.numel() for t in W.values())
    w_bytes = n_params * 2

    # Prefill：一次性处理长 prompt
    long_ids = torch.randint(0, 150000, (1, 1024), device=device)
    mine.generate(long_ids[:, :32], 4)   # 预热
    _, t_pf, dec = mine.generate(long_ids, 32)
    flops = 2 * n_params * 1024
    print(f"   Prefill 1024 个 token: {t_pf * 1e3:.1f} ms → {1024 / t_pf:,.0f} token/s，"
          f"约 {flops / t_pf / 1e12:.1f} TFLOP/s（线性层部分，按 2N FLOPs/token 估算）")
    per_tok = sum(dec) / len(dec)
    bound = w_bytes / (bw_gbs * 1e9)
    print(f"   Decode（上下文 ~1K）: {per_tok * 1e3:.2f} ms/token → {1 / per_tok:.0f} token/s")
    print(f"   带宽下界: 权重 {w_bytes / 1e9:.2f} GB / {bw_gbs:.0f} GB/s = {bound * 1e3:.2f} ms/token（{1 / bound:.0f} token/s 上限）")
    print(f"   👉 我们只达到了上限的 {bound / per_tok:.0%}。差距来自：24 层 × 每层十几个小 kernel 的启动开销、")
    print("      未融合的逐元素算子反复读写显存、repeat_interleave 复制 KV 等。vLLM / llama.cpp 用算子融合、")
    print("      CUDA Graph、专用 GEMV 与注意力 kernel 把这些开销吃掉——这正是推理引擎存在的意义（capstone_engines_practice）。")

    print("\n   朴素（每步重算整段）vs KV Cache，在不同上下文长度下每生成 1 个 token 的耗时:")
    print(f"   {'上下文':>8} | {'朴素 ms':>9} | {'KV Cache ms':>12} | {'倍数':>6}")
    for ctx in [128, 512, 1024, 2048]:
        ids = torch.randint(0, 150000, (1, ctx), device=device)
        _, _, d_naive = mine.generate(ids, 6, use_cache=False)
        _, _, d_cache = mine.generate(ids, 6, use_cache=True)
        a, b = sum(d_naive[1:]) / len(d_naive[1:]), sum(d_cache[1:]) / len(d_cache[1:])
        print(f"   {ctx:>8} | {a * 1e3:9.2f} | {b * 1e3:12.2f} | {a / b:5.1f}x")
    print("   👉 与 Lab 05 的迷你模型不同，真实模型上重算的代价在 GPU 上也清清楚楚：上下文越长差距越大。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bandwidth", type=float, default=380.0, help="你的显卡实测带宽 GB/s（见 measure_gpu_roofline.py）")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    model_dir = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(model_dir)

    cfg = exp1_config(model_dir)
    W = exp2_weights(model_dir, cfg)
    exp3_tokenizer(tok)
    ref, mine = exp4_logits_match(model_dir, tok, W, cfg, args.device)
    exp5_generate_match(tok, ref, mine, args.device)
    del ref, mine
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        exp6_perf(W, cfg, tok, args.device, args.bandwidth)


if __name__ == "__main__":
    main()
