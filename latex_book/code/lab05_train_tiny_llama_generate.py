"""
Lab 05 - 推理: 在一个"真正训练过"的模型上，把第 6、7 篇的推理理论全部跑一遍

实验清单:
  1. 朴素自回归（每步重算整段）vs KV Cache（Prefill 一次 + 每步只算 1 个 token）
     → 逐 token 比对输出完全一致，logits 误差在浮点噪声量级
  2. 用真实缓存张量验证第 6 篇的 KV Cache 显存公式 2 × L × n_kv × d_head × bytes × tokens
  3. 每一步 Decode 的耗时：朴素方式随长度增长，缓存方式基本持平
  4. 采样策略：Greedy / Temperature / Top-P 在真实模型上的效果（第 7 篇）

运行（先跑 train.py 生成 tiny_llama.pt）:
  .venv/bin/python lab05_train_tiny_llama/generate.py
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from model import Config, TinyLlama

HERE = os.path.dirname(os.path.abspath(__file__))


def load(device):
    path = os.path.join(HERE, "tiny_llama.pt")
    if not os.path.exists(path):
        raise SystemExit("找不到 tiny_llama.pt，请先运行: .venv/bin/python lab05_train_tiny_llama/train.py")
    ckpt = torch.load(path, map_location=device)
    model = TinyLlama(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    chars = ckpt["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    return model, chars, stoi


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def generate_naive(model, ids, n_new, device):
    """朴素版：每一步都把【整段序列】重新送进模型，只取最后一个位置的 logits"""
    ids = list(ids)
    step_logits, step_times = [], []
    for _ in range(n_new):
        sync(device); t = time.perf_counter()
        x = torch.tensor([ids], device=device)
        logits = model(x)[0, -1]
        nxt = int(logits.argmax())
        sync(device); step_times.append(time.perf_counter() - t)
        step_logits.append(logits.float().cpu())
        ids.append(nxt)
    return ids, step_logits, step_times


@torch.inference_mode()
def generate_cached(model, ids, n_new, device):
    """KV Cache 版：Prefill 整段 prompt 一次，之后每步只送 1 个 token 进去"""
    ids = list(ids)
    cache = model.new_cache()
    step_logits, step_times = [], []

    sync(device); t = time.perf_counter()
    logits = model(torch.tensor([ids], device=device), cache=cache, start_pos=0)[0, -1]   # Prefill
    for i in range(n_new):
        nxt = int(logits.argmax())
        sync(device); step_times.append(time.perf_counter() - t)
        step_logits.append(logits.float().cpu())
        ids.append(nxt)
        if i == n_new - 1:
            break
        sync(device); t = time.perf_counter()
        # Decode：只输入刚生成的 1 个 token，它的位置是 len(ids) - 1
        logits = model(torch.tensor([[nxt]], device=device), cache=cache, start_pos=len(ids) - 1)[0, -1]
    return ids, step_logits, step_times, cache


@torch.inference_mode()
def sample(model, ids, n_new, device, temperature=1.0, top_p=1.0, seed=0):
    """带温度与 Top-P 的采样（第 7 篇），使用 KV Cache"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = list(ids)
    cache = model.new_cache()
    logits = model(torch.tensor([ids], device=device), cache=cache, start_pos=0)[0, -1]
    for _ in range(n_new):
        logits = logits.float().cpu()
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sp, si = probs.sort(descending=True)
                keep = sp.cumsum(0) - sp < top_p        # 累计概率"在加上自己之前"还没到 P 的都保留
                sp = sp * keep
                probs = torch.zeros_like(probs).scatter(0, si, sp)
                probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1, generator=g))
        ids.append(nxt)
        logits = model(torch.tensor([[nxt]], device=device), cache=cache, start_pos=len(ids) - 1)[0, -1]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="KV Cache 的")
    ap.add_argument("--n", type=int, default=400, help="生成多少个字符")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    model, chars, stoi = load(dev)
    cfg = model.cfg
    decode = lambda ids: "".join(chars[i] for i in ids)
    prompt_ids = [stoi[c] for c in args.prompt if c in stoi]
    n = min(args.n, cfg.max_seq_len - len(prompt_ids))
    print(f"[决策] 已加载模型: {model.num_params() / 1e6:.2f}M 参数 | 设备 {dev} | 提示词 {args.prompt!r}")
    generate_cached(model, prompt_ids, 8, dev)   # 预热：第一次调用 CUDA kernel 有初始化开销，不计入计时

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[实验] 实验 1: 朴素自回归 vs KV Cache —— 输出必须完全一致")
    print("=" * 76)
    ids_a, logits_a, times_a = generate_naive(model, prompt_ids, n, dev)
    ids_b, logits_b, times_b, cache = generate_cached(model, prompt_ids, n, dev)
    same = ids_a == ids_b
    max_diff = max((a - b).abs().max().item() for a, b in zip(logits_a, logits_b))
    print(f"   生成 {n} 个字符，两种方式 token 序列{'完全相同 [OK]' if same else '不同 [X]'}；各步 logits 最大差值 {max_diff:.2e}")
    assert same and max_diff < 1e-3, 'KV Cache 与朴素生成结果不一致'
    print("   （差值不为 0 是因为矩阵形状不同时浮点加法的顺序不同，属于正常的数值噪声）")
    print("\n   Greedy 生成结果（前 150 字）:")
    print("   " + decode(ids_b)[:150].replace("\n", "\n   "))

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[实验] 实验 2: 用真实缓存张量验证第 6 篇的 KV Cache 显存公式")
    print("=" * 76)
    tokens_cached = cache[0]["k"].shape[2]
    actual = sum(t.numel() * t.element_size() for layer in cache for t in layer.values())
    elem = cache[0]["k"].element_size()
    formula = 2 * cfg.n_layer * cfg.n_kv_head * cfg.head_dim * elem * tokens_cached
    print(f"   缓存中 token 数 = {tokens_cached}，每层 K 的形状 = {list(cache[0]['k'].shape)}  ([batch, n_kv_head, tokens, head_dim])")
    print(f"   公式 2 × L({cfg.n_layer}) × n_kv({cfg.n_kv_head}) × d_head({cfg.head_dim}) × {elem}B × {tokens_cached} = {formula:,} 字节")
    print(f"   实际缓存张量占用                                   = {actual:,} 字节  {'[OK]' if formula == actual else '[X]'}")
    assert formula == actual, 'KV Cache 显存公式与实际不符'
    mha = formula * cfg.n_head // cfg.n_kv_head
    print(f"   如果不用 GQA（MHA，{cfg.n_head} 个 KV 头），会是 {mha:,} 字节，是现在的 {cfg.n_head // cfg.n_kv_head} 倍")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[实验] 实验 3: 每一步的耗时（毫秒）")
    print("=" * 76)
    results = {dev: (times_a, times_b)}
    if dev != "cpu":
        # 再在 CPU 上跑一遍：CPU 没有"kernel 启动开销被 GPU 并行掩盖"的问题，重算的代价会直接暴露出来
        model_cpu, _, _ = load("cpu")
        _, _, ta = generate_naive(model_cpu, prompt_ids, n, "cpu")
        _, _, tb, _ = generate_cached(model_cpu, prompt_ids, n, "cpu")
        results["cpu"] = (ta, tb)
    header = " | ".join(f"{'朴素(' + d + ')':>12} | {'Cache(' + d + ')':>12}" for d in results)
    print(f"   {'第几步':>6} | {'上下文长度':>8} | {header}")
    for i in [1, n // 4, n // 2, 3 * n // 4, n - 1]:
        row = " | ".join(f"{ta[i] * 1e3:12.2f} | {tb[i] * 1e3:12.2f}" for ta, tb in results.values())
        print(f"   {i:>6} | {len(prompt_ids) + i:>8} | {row}")
    for d, (ta, tb) in results.items():
        print(f"   [{d}] 总耗时: 朴素 {sum(ta) * 1e3:.0f} ms vs KV Cache {sum(tb) * 1e3:.0f} ms")
    print("   --> 解读：")
    print("      - CPU 上：朴素方式每步耗时随上下文线性增长（每步都重算整段），KV Cache 基本持平 —— 第 6 篇的结论。")
    print("      - GPU 上：两者可能差不多！因为这个模型只有几 MB，每步真正的计算只要几微秒，耗时几乎全是")
    print("        Python 与 kernel 启动的固定开销（第 8 篇引用的 Horace He 文章称之为 overhead），多算一些也看不出来。")
    print("        模型越大、上下文越长，计算才会超过固定开销；Lab 07b 在真实的 0.5B 模型上会看到明显差距。")
    print("        这也是 vLLM 等引擎要用 CUDA Graph 把整步 Decode 录制成一次启动的原因。")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[实验] 实验 4: 采样策略对比（第 7 篇），每种生成 80 个字符")
    print("=" * 76)
    for name, kw in [("Greedy (T=0)", dict(temperature=0)),
                     ("T=0.8, Top-P=0.9", dict(temperature=0.8, top_p=0.9)),
                     ("T=1.5 (高温)", dict(temperature=1.5)),
                     ("T=3.0 (极高温)", dict(temperature=3.0))]:
        out = decode(sample(model, prompt_ids, 80, dev, **kw))
        print(f"   [{name:<16}] {out.replace(chr(10), ' ')}")
    print("   --> 温度越高越'有创意'，极高温时退化成接近均匀的随机字符（第 7 篇定理 2）。")


if __name__ == "__main__":
    main()
