"""
Lab 09: 用 NumPy 从零实现 Online Softmax、FlashAttention 与 FlashDecoding，并逐一验证与朴素注意力完全相等

对应教程: zero_to_hero_tutorial/09_FlashAttention：OnlineSoftmax的严格推导与IO复杂度.md

实验清单:
  1. Online Softmax：一遍扫描得到的 (m, ℓ) 与三遍扫描完全一致；逐步打印修正因子
  2. 分块 FlashAttention（含因果掩码与整块跳过）vs 朴素注意力：误差 ~1e-15
  3. 显存访问量模型：朴素 Θ(Nd + N²) vs FlashAttention Θ(N²d²/M)
  4. FlashDecoding：把 KV 切成若干段各自计算，再用合并公式合并，结果与整体计算一致；合并顺序无关（结合律）
  5. （可选，需要 PyTorch + GPU）朴素注意力 vs F.scaled_dot_product_attention 的真实显存与耗时

运行:
  .venv/bin/python lab09_flash_attention/flash_attention_numpy.py
"""

import math

import numpy as np


def title(s):
    print("\n" + "=" * 78)
    print(f"🔬 {s}")
    print("=" * 78)


def naive_attention(Q, K, V, causal=False):
    """朴素实现：显式构造 N×N 的 S 和 P"""
    S = Q @ K.T / math.sqrt(Q.shape[1])
    if causal:
        S = S + np.triu(np.full(S.shape, -np.inf), k=1)
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=1, keepdims=True)
    return P @ V


# =====================================================================
# 实验 1: Online Softmax
# =====================================================================
def exp1_online_softmax():
    title("实验 1: Online Softmax —— 一遍扫描求出最大值 m 与归一化因子 ℓ")
    x = np.array([1.0, 3.0, 2.0, 8.0, 5.0, 9.0, 4.0])
    m, l = -np.inf, 0.0
    print(f"   输入 x = {x.tolist()}")
    print(f"   {'j':>3} | {'x_j':>5} | {'m_j':>5} | {'修正因子 e^(m_old - m_new)':>26} | {'ℓ_j':>10}")
    for j, xj in enumerate(x, 1):
        m_new = max(m, xj)
        corr = math.exp(m - m_new) if m > -np.inf else 0.0
        l = l * corr + math.exp(xj - m_new)
        m = m_new
        print(f"   {j:>3} | {xj:5.1f} | {m:5.1f} | {corr:26.6f} | {l:10.6f}")
    m_ref = x.max()
    l_ref = np.exp(x - m_ref).sum()
    print(f"   三遍扫描的标准答案: m = {m_ref}, ℓ = {l_ref:.6f}  → 一致 ✅" if abs(l - l_ref) < 1e-12 else "   ❌ 不一致")
    assert abs(l - l_ref) < 1e-12 and m == m_ref
    print("   👉 每当出现新的最大值（第 2、4、6 步），之前的和就乘上一个 ≤ 1 的修正因子，换算到新基准。")


# =====================================================================
# 实验 2: 分块 FlashAttention
# =====================================================================
def flash_attention(Q, K, V, Br=16, Bc=16, causal=False, stats=None):
    """
    FlashAttention-2 的循环顺序（外层 Q 块，内层 K/V 块）。
    所有 Br×Bc 的中间矩阵都是局部变量（对应 GPU 上的 SRAM / 寄存器），从不构造 N×N 矩阵。
    stats: 传入 dict 时，统计"从显存读写的元素个数"和"跳过的块数"
    """
    N, d = Q.shape
    O = np.zeros_like(Q)
    scale = 1.0 / math.sqrt(d)
    for i0 in range(0, N, Br):
        Qi = Q[i0:i0 + Br]                        # 读 Q 块
        br = Qi.shape[0]
        m = np.full(br, -np.inf)
        l = np.zeros(br)
        acc = np.zeros((br, d))
        if stats is not None:
            stats["hbm"] += Qi.size
        for j0 in range(0, N, Bc):
            if causal and j0 > i0 + br - 1:      # 整块都在未来 → 直接跳过（因果注意力省一半计算）
                if stats is not None:
                    stats["skipped"] += 1
                continue
            Kj, Vj = K[j0:j0 + Bc], V[j0:j0 + Bc]  # 读 K、V 块
            if stats is not None:
                stats["hbm"] += Kj.size + Vj.size
                stats["blocks"] += 1
            S = Qi @ Kj.T * scale                  # [br, bc]，只存在"片上"
            if causal:
                q_pos = np.arange(i0, i0 + br)[:, None]
                k_pos = np.arange(j0, j0 + Kj.shape[0])[None, :]
                S = np.where(k_pos > q_pos, -np.inf, S)
            m_new = np.maximum(m, S.max(axis=1))
            P = np.exp(S - m_new[:, None])
            corr = np.exp(m - m_new)               # 修正因子
            l = l * corr + P.sum(axis=1)
            acc = acc * corr[:, None] + P @ Vj
            m = m_new
        O[i0:i0 + br] = acc / l[:, None]           # 写 O 块
        if stats is not None:
            stats["hbm"] += Qi.size
    return O


def exp2_flash_vs_naive():
    title("实验 2: 分块 FlashAttention vs 朴素注意力（数学上严格相等）")
    rng = np.random.default_rng(0)
    N, d = 256, 64
    Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
    for causal in [False, True]:
        ref = naive_attention(Q, K, V, causal)
        for Br, Bc in [(16, 16), (32, 64), (64, 32), (256, 256)]:
            st = {"hbm": 0, "blocks": 0, "skipped": 0}
            out = flash_attention(Q, K, V, Br, Bc, causal, st)
            print(f"   {'因果' if causal else '双向'} | 块大小 Br={Br:<3} Bc={Bc:<3} | 最大误差 {np.abs(out - ref).max():.1e} | "
                  f"计算 {st['blocks']:3d} 块，跳过 {st['skipped']:3d} 块")
            assert np.abs(out - ref).max() < 1e-10, 'FlashAttention 与朴素注意力不一致'
    print("   👉 无论块怎么切，结果都与朴素实现相等（误差是 1e-15 的浮点舍入）；因果注意力约一半的块被整块跳过。")


# =====================================================================
# 实验 3: 显存访问量模型
# =====================================================================
def exp3_io_model():
    title("实验 3: 显存访问量（元素个数）—— 朴素 vs FlashAttention")
    sram_elems = 100 * 1024 // 2      # 约 100 KB 片上 SRAM，BF16
    print("   假设片上 SRAM ≈ 100 KB。朴素 = 读 Q,K,V + 写 O + (写 S、读 S、写 P、读 P)；")
    print("   Flash = 读 Q + 写 O + 每个 Q 块都把 K、V 完整读一遍（共 ceil(N/Br) 遍）")
    print("   本文件用的是【元素个数】口径，不是字节。")
    for d in [64, 128]:
        Br = max(16, sram_elems // (4 * d) // 16 * 16)   # Q、K、V、O 四个块大致放得下
        # 本模型：朴素 = 4Nd + 4N²，Flash = 2Nd + 2N²d/Br
        # N→∞ 时比值 → 2·Br/d，代入 Br ≈ M/(4d) 得 M/(2d²)。
        ratio_pred = 2 * Br / d
        print(f"\n   头维度 d = {d} → 块大小约 Br = Bc = {Br}（Br ≈ M/(4d)，M = {sram_elems} 元素）")
        print(f"   本模型的渐近节省倍数 = 2·Br/d = {ratio_pred:.1f}，"
              f"等价地 M/(2d²) = {sram_elems / (2 * d * d):.1f}")
        print(f"   论文写的是 M/d² = {sram_elems / (d * d):.1f} —— 是同一个量级，差的这个 2 来自记账口径，")
        print(f"   不是矛盾：本文的朴素项数了 4 趟 N²（写 S/读 S/写 P/读 P），论文的 Θ(Nd + N²) 只按元素数算一趟。")
        print(f"   {'序列长度 N':>10} | {'朴素':>10} | {'Flash':>10} | {'节省':>6} | {'朴素需要的 N² 额外显存 (1 个头)':>28}")
        ratios = []
        for N in [1024, 4096, 16384, 65536]:
            naive = 4 * N * d + 4 * N * N
            n_kv_passes = -(-N // Br)                     # ceil，不是 floor：最后那个不满的块也要读
            flash = 2 * N * d + n_kv_passes * 2 * N * d
            ratios.append(naive / flash)
            print(f"   {N:>10} | {naive:>10.2e} | {flash:>10.2e} | {naive / flash:5.1f}x | {2 * N * N * 2 / 1024**2:>25.0f} MB")
        # 断言：比值单调趋近 2·Br/d，且永远不超过它（尾块只会让节省变少，不会变多）
        assert all(ratios[i] <= ratios[i + 1] + 1e-9 for i in range(len(ratios) - 1)), \
            f"节省倍数应随 N 单调上升（尾块占比下降），实际 {ratios}"
        assert ratios[-1] <= ratio_pred + 1e-9, \
            f"节省倍数 {ratios[-1]:.2f} 不该超过渐近值 {ratio_pred:.2f}（ceil 只可能让它更小）"
        assert ratios[-1] > ratio_pred * 0.95, \
            f"N=65536 时应已接近渐近值 {ratio_pred:.2f}，实际 {ratios[-1]:.2f}"
    print("\n   👉 注意：访存节省倍数是一个【与 N 无关的常数】（d 越小、SRAM 越大，省得越多），并不会随 N 无限增大。")
    print(f"      但小 N 时它还到不了那个常数（尾块占比大）；本例要到 N≳16K 才收敛到 {ratio_pred:.1f}x。")
    print("      随 N 增长的真正收益是【额外显存从 O(N²) 降到 O(N)】—— 没有它，长上下文根本放不进显存。")
    print("      再加上多个 kernel 融合成一个（省掉启动开销与 FP32 中间结果），实际加速往往比这个模型更大，见实验 5。")
    print("      ⚠️ 这条结论在真机上不总是成立：把整行缓存进 shared 会压低占用率，N 一大反而变慢 ——")
    print("         见 cuda_softmax/README.md 里 N=8192 的实测（同一个道理也是 FlashAttention 必须分块的原因）。")


# =====================================================================
# 实验 4: FlashDecoding
# =====================================================================
def partial_attention(q, K, V):
    """对一段 KV 计算部分结果 (m, ℓ, acc)，q: [d]"""
    s = K @ q / math.sqrt(q.shape[0])
    m = s.max()
    p = np.exp(s - m)
    return m, p.sum(), p @ V


def merge(parts):
    """第 9 篇 §4.2 的合并公式"""
    m = max(p[0] for p in parts)
    l = sum(p[1] * math.exp(p[0] - m) for p in parts)
    acc = sum(p[2] * math.exp(p[0] - m) for p in parts)
    return m, l, acc


def exp4_flash_decoding():
    title("实验 4: FlashDecoding —— 切分 KV 并行计算，再合并")
    rng = np.random.default_rng(1)
    S, d = 8192, 128
    q = rng.standard_normal(d)
    K, V = rng.standard_normal((S, d)), rng.standard_normal((S, d))
    ref = naive_attention(q[None, :], K, V)[0]
    for n_split in [1, 4, 16, 64]:
        chunks = np.array_split(np.arange(S), n_split)
        parts = [partial_attention(q, K[c], V[c]) for c in chunks]    # 在 GPU 上，这些段分给不同的 SM 同时算
        m, l, acc = merge(parts)
        print(f"   切成 {n_split:>2} 段并行 → 合并后最大误差 {np.abs(acc / l - ref).max():.1e}")
        assert np.abs(acc / l - ref).max() < 1e-10
    parts = [partial_attention(q, K[c], V[c]) for c in np.array_split(np.arange(S), 8)]
    a = merge([merge(parts[:3]), merge(parts[3:])])
    b = merge([merge(parts[::2]), merge(parts[1::2])])
    print(f"   结合律：两种不同的分组合并顺序，结果差 {np.abs(a[2] / a[1] - b[2] / b[1]).max():.1e}")
    assert np.abs(a[2] / a[1] - b[2] / b[1]).max() < 1e-12
    print("   👉 合并满足结合律，所以可以任意切分、任意顺序合并 —— 这也是 PagedAttention 和跨卡 Ring Attention 的基础。")


# =====================================================================
# 实验 5（可选）: GPU 上的真实差距
# =====================================================================
def exp5_gpu():
    title("实验 5（可选）: GPU 上朴素注意力 vs PyTorch SDPA（内部调用 FlashAttention 类 kernel）")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("   未安装 PyTorch，跳过。")
        return
    if not torch.cuda.is_available():
        print("   没有 CUDA，跳过。")
        return

    def naive(q, k, v):
        s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        n = s.shape[-1]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
        return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v

    def measure(fn, *a):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        for _ in range(2):
            fn(*a)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(5):
            fn(*a)
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) / 5, (torch.cuda.max_memory_allocated() - base) / 1024**2

    H, d = 8, 128
    print(f"   单层、{H} 个头、d={d}、BF16、因果注意力")
    print(f"   {'N':>6} | {'朴素 ms':>8} | {'朴素额外显存':>12} | {'SDPA ms':>8} | {'SDPA 额外显存':>13}")
    for N in [1024, 4096, 8192, 16384]:
        q, k, v = (torch.randn(1, H, N, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ts, ms = measure(sdpa, q, k, v)
        try:
            tn, mn = measure(naive, q, k, v)
            naive_str = f"{tn:8.2f} | {mn:9.0f} MB"
        except torch.OutOfMemoryError:
            naive_str = f"{'OOM':>8} | {'显存不足':>10}"
            torch.cuda.empty_cache()
        print(f"   {N:>6} | {naive_str} | {ts:8.2f} | {ms:10.0f} MB")
        del q, k, v
        torch.cuda.empty_cache()
    print("   👉 朴素实现的额外显存随 N² 增长，SDPA 基本只和输出一样大；速度差距也随 N 拉大。")
    print("      朴素实现还有好几个 N² 的中间张量（FP32 的 softmax、掩码矩阵），所以实际加速比实验 3 的访存模型更大。")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"   ⚠️ 如果某一行朴素实现的'额外显存'超过了你的显存容量（{total:.0f} MB）却没有 OOM、而是慢了上百倍：")
    print("      这是 Windows / WSL 的显卡驱动把放不下的部分挪到了系统内存（sysmem fallback），经过 PCIe 访问，")
    print("      带宽只有显存的几十分之一。在 Linux 原生驱动或数据中心卡上，这里会直接 OOM。")


if __name__ == "__main__":
    exp1_online_softmax()
    exp2_flash_vs_naive()
    exp3_io_model()
    exp4_flash_decoding()
    exp5_gpu()
