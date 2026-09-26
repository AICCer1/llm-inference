"""
Lab 13: 用 NumPy 模拟多卡并行推理 —— 张量并行、Ring All-Reduce 与多卡 Decode 延迟模型

对应教程: zero_to_hero_tutorial/13_多卡并行推理：张量并行、流水线并行与专家并行.md

实验清单:
  1. 张量并行 SwiGLU：gate/up 按列切、down 按行切，一次 All-Reduce 后与单卡结果一致
  2. 反例：如果 gate/up 按行切（切输入维度），在部分和上直接做 SiLU 会算错
  3. 张量并行注意力：按头切分，每张"卡"只存自己那几个头的 KV，o_proj 按行切，一次 All-Reduce
  4. Ring All-Reduce：Reduce-Scatter + All-Gather，每张卡发送 2(n-1)/n 倍数据
  5. 多卡 Decode 延迟模型：Llama-3-70B，TP=1/2/4/8，NVLink vs PCIe；以及流水线并行为什么不降延迟

运行:
  .venv/bin/python lab13_parallelism/tensor_parallel_sim.py
"""

import math

import numpy as np

rng = np.random.default_rng(0)


def title(s):
    print("\n" + "=" * 78)
    print(f"[实验] {s}")
    print("=" * 78)


def silu(x):
    return x / (1 + np.exp(-x))


def all_reduce(parts):
    """模拟 All-Reduce：所有卡的张量求和，每张卡都拿到结果"""
    total = sum(parts)
    return [total.copy() for _ in parts]


# =====================================================================
# 实验 1 & 2
# =====================================================================
def exp1_2_mlp():
    T, d, h = 4, 64, 176
    X = rng.standard_normal((T, d))
    Wg, Wu = rng.standard_normal((d, h)) * 0.1, rng.standard_normal((d, h)) * 0.1
    Wd = rng.standard_normal((h, d)) * 0.1
    Y_ref = (silu(X @ Wg) * (X @ Wu)) @ Wd

    title("实验 1: 张量并行 SwiGLU —— 先按列切，再按行切，只需一次 All-Reduce")
    for n in [2, 4, 8]:
        cols = np.array_split(np.arange(h), n)
        partial = []
        for c in cols:                                         # 每张卡只持有 1/n 的权重
            local = silu(X @ Wg[:, c]) * (X @ Wu[:, c])        # 本地完成非线性，无需通信
            partial.append(local @ Wd[c, :])                   # 得到一个部分和 [T, d]
        Y = all_reduce(partial)[0]
        print(f"   TP={n}: 每卡权重 {Wg[:, cols[0]].size * 3} 个参数（单卡 {Wg.size * 3}），与单卡结果最大误差 {np.abs(Y - Y_ref).max():.1e}")
        assert np.abs(Y - Y_ref).max() < 1e-12

    title("实验 2: 反例 —— gate/up 按行切（切输入维度），在部分和上做 SiLU")
    n = 2
    rows = np.array_split(np.arange(d), n)
    wrong = sum((silu(X[:, r] @ Wg[r]) * (X[:, r] @ Wu[r])) @ Wd for r in rows)
    print(f"   错误做法 Σ f(X_i A_i) 与正确结果的最大误差: {np.abs(wrong - Y_ref).max():.3f}（相对 {np.abs(wrong - Y_ref).max() / np.abs(Y_ref).max():.0%}）")
    assert np.abs(wrong - Y_ref).max() > 0.1, '错误切法应产生明显误差'
    g = sum(X[:, r] @ Wg[r] for r in rows)                   # 必须先 All-Reduce 出完整的 XA
    u = sum(X[:, r] @ Wu[r] for r in rows)
    right = (silu(g) * u) @ Wd
    print(f"   正确做法（先 All-Reduce 再做 SiLU）误差: {np.abs(right - Y_ref).max():.1e} —— 但多了一次通信")
    print("   --> 非线性对加法不分配：f(a+b) ≠ f(a)+f(b)。'先列后行'让非线性在本地完成，这是 Megatron 切法的精髓。")


# =====================================================================
# 实验 3
# =====================================================================
def attention(Q, K, V):
    s = Q @ K.T / math.sqrt(Q.shape[1])
    s = s + np.triu(np.full(s.shape, -np.inf), 1)
    p = np.exp(s - s.max(1, keepdims=True))
    return (p / p.sum(1, keepdims=True)) @ V


def exp3_attention():
    title("实验 3: 张量并行注意力 —— 按头切分")
    T, d, n_h, hd = 6, 64, 8, 8
    X = rng.standard_normal((T, d))
    Wq, Wk, Wv = (rng.standard_normal((d, n_h * hd)) * 0.1 for _ in range(3))
    Wo = rng.standard_normal((n_h * hd, d)) * 0.1

    def heads_out(hs):
        outs = []
        for h in hs:
            c = slice(h * hd, (h + 1) * hd)
            outs.append(attention(X @ Wq[:, c], X @ Wk[:, c], X @ Wv[:, c]))
        return np.concatenate(outs, axis=1)

    Y_ref = heads_out(range(n_h)) @ Wo
    for n in [2, 4]:
        groups = np.array_split(np.arange(n_h), n)
        partial = []
        for g in groups:
            cols = np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in g])
            partial.append(heads_out(g) @ Wo[cols, :])           # o_proj 按行切
        Y = all_reduce(partial)[0]
        print(f"   TP={n}: 每卡 {len(groups[0])} 个头、只存 1/{n} 的 KV Cache，与单卡最大误差 {np.abs(Y - Y_ref).max():.1e}")
        assert np.abs(Y - Y_ref).max() < 1e-12
    print("   --> 一层 Transformer = 注意力 1 次 All-Reduce + FFN 1 次 All-Reduce，共 2 次。")


# =====================================================================
# 实验 4
# =====================================================================
def ring_all_reduce(tensors):
    """
    Ring All-Reduce：n 张卡排成环，数据切成 n 块。
    阶段 1 Reduce-Scatter（n-1 步）：每步把一块发给右邻居并累加 → 结束时卡 i 拥有第 (i+1)%n 块的完整和
    阶段 2 All-Gather（n-1 步）：每步把完整的块传给右邻居 → 结束时人人拥有全部块
    """
    n = len(tensors)
    chunks = [np.array_split(t.copy(), n) for t in tensors]
    sent = [0] * n
    for step in range(n - 1):                                   # Reduce-Scatter
        msgs = [(i, (i - step) % n, chunks[i][(i - step) % n].copy()) for i in range(n)]
        for i, c, data in msgs:
            chunks[(i + 1) % n][c] += data
            sent[i] += data.size
    for step in range(n - 1):                                   # All-Gather
        msgs = [(i, (i + 1 - step) % n, chunks[i][(i + 1 - step) % n].copy()) for i in range(n)]
        for i, c, data in msgs:
            chunks[(i + 1) % n][c] = data
            sent[i] += data.size
    return [np.concatenate(c) for c in chunks], sent


def exp4_ring():
    title("实验 4: Ring All-Reduce 的正确性与通信量")
    size = 1 << 16
    for n in [2, 4, 8]:
        ts = [rng.standard_normal(size) for _ in range(n)]
        out, sent = ring_all_reduce(ts)
        ref = sum(ts)
        err = max(np.abs(o - ref).max() for o in out)
        print(f"   {n} 张卡: 结果误差 {err:.1e} | 每卡发送 {sent[0] / size:.3f} × 数据量，理论 2(n-1)/n = {2 * (n - 1) / n:.3f}")
        assert err < 1e-12 and abs(sent[0] / size - 2 * (n - 1) / n) < 1e-9
    print("   --> 卡数从 2 到 8，每卡发送量只从 1.0 倍涨到 1.75 倍，上限 2 倍 —— 带宽代价几乎与卡数无关。")


# =====================================================================
# 实验 5
# =====================================================================
def exp5_latency_model():
    title("实验 5: 多卡 Decode 延迟模型 —— Llama-3-70B BF16，batch=1")
    L, d, params = 80, 8192, 70.6e9
    hbm_bw = 3.35e12                                           # H100 SXM 显存带宽
    # 每次 All-Reduce 的固定延迟是粗略假设（NVLink 约 5 µs、PCIe 约 20 µs），真实值取决于拓扑与 NCCL 版本
    links = {"NVLink (H100, ~450 GB/s)": (450e9, 5e-6), "PCIe 5.0 (~50 GB/s)": (50e9, 20e-6)}
    msg = 1 * d * 2                                            # 每次 All-Reduce：1 个 token × d × 2 字节
    print(f"   每 token: 读权重 {params * 2 / 1e9:.0f} GB；每步 {2 * L} 次 All-Reduce，每次仅 {msg / 1024:.0f} KB（很小 → 由固定延迟主导）")
    print(f"   {'互联':<24} | {'TP':>3} | {'读权重 ms':>9} | {'通信 ms':>8} | {'每 token ms':>11} | {'相对 TP=1':>9}")
    speedups = {}
    for name, (bw, lat) in links.items():
        base = None
        for n in [1, 2, 4, 8]:
            t_mem = params * 2 / n / hbm_bw
            t_comm = 0 if n == 1 else 2 * L * (lat + 2 * (n - 1) / n * msg / bw)
            t = t_mem + t_comm
            base = base or t
            speedups[(name, n)] = base / t
            print(f"   {name:<22} | {n:>3} | {t_mem * 1e3:9.2f} | {t_comm * 1e3:8.2f} | {t * 1e3:11.2f} | {base / t:8.2f}x")
    # 加速比必须是次线性的：通信吃掉一部分收益，而且卡越多占比越高
    for name in links:
        sp = [speedups[(name, n)] for n in (1, 2, 4, 8)]
        assert sp[0] == 1.0, f"{name}: TP=1 的加速比应为 1，实际 {sp[0]}"
        assert all(sp[i] < sp[i + 1] for i in range(3)), f"{name}: 加速比应随 n 单调递增，实际 {sp}"
        assert sp[3] < 8.0, f"{name}: TP=8 的加速比 {sp[3]:.2f} 不该达到理想的 8 倍（通信不可能免费）"
    # NVLink 的通信带宽远高于 PCIe，所以它的扩展效率必须更好
    nv = next(v for (name, n), v in speedups.items() if name.startswith("NVLink") and n == 8)
    pc = next(v for (name, n), v in speedups.items() if name.startswith("PCIe") and n == 8)
    assert nv > pc * 1.2, f"NVLink 上 TP=8 的加速比 {nv:.2f} 应显著优于 PCIe 的 {pc:.2f}"
    print("   （TP=1 需要 141 GB 显存，单张 80GB 卡其实放不下，这里只作为计算基准）")
    print("   --> NVLink 上 TP=8 仍有约 7 倍加速；PCIe 上通信的固定延迟每 token 要累加 160 次，TP=8 时占总耗时近 40%，")
    print("      扩展效率明显下降。而且通信时间几乎不随 n 减少，卡越多占比越高 —— 所以 TP 通常只在 NVLink 连接的机内使用。")

    print("\n   流水线并行 PP=4（每卡 20 层）：")
    t_pp = params * 2 / hbm_bw + 3 * 20e-6                    # 各段依次执行，总读取量不变，外加 3 次段间传输
    t_single = params * 2 / hbm_bw
    # PP 的要点：延迟几乎不变（只多了几次段间传输），换的是吞吐
    assert t_pp > t_single and t_pp < t_single * 1.5, \
        f"PP 的延迟应接近单卡（{t_single * 1e3:.2f} ms），实际 {t_pp * 1e3:.2f} ms"
    print(f"   单请求每 token ≈ {t_pp * 1e3:.2f} ms —— 与单卡的 {t_single * 1e3:.2f} ms 基本相同，"
          f"PP 不降低延迟；但 4 段可以同时处理 4 批不同的请求，吞吐 ×4。")


if __name__ == "__main__":
    exp1_2_mlp()
    exp3_attention()
    exp4_ring()
    exp5_latency_model()
