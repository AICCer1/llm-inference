"""
Lab 08 - 实测脚本: 在你自己的 GPU 上测出 Roofline，并验证第 8 篇的三个推导 (需要 PyTorch + CUDA)

对应教程: zero_to_hero_tutorial/08_硬件视角：GPU存储层级、FLOPs、算术强度与浮点格式.md

实验清单:
  1. 显卡基本信息 (SM 数、显存、L2 大小)
  2. 实测显存带宽 β (大张量拷贝)
  3. 实测 BF16 Tensor Core 峰值算力 π (大方阵乘)，得到你这张卡的拐点 AI* = π / β
  4. 【验证推导 1】Decode 形状的线性层 [B, K] x [K, N]: batch 从 1 涨到 1024，
     耗时在 B 远小于拐点时几乎不变 → 算术强度 ≈ B
  5. 【验证推导 2】Decode 注意力: GQA 分组大小 g 从 1 涨到 8，读取的 KV 字节不变、FLOPs 涨 g 倍，
     耗时几乎不变 → 算术强度 ≈ g，与 batch 无关
  6. 【验证第 8 篇 §6】BF16 / FP16 的精度与范围陷阱 (CPU 即可运行)

运行:
  source .venv/bin/activate
  python lab08_memory_and_roofline/measure_gpu_roofline.py
"""

import torch


def title(s):
    print("\n" + "=" * 78)
    print(f"[实验] {s}")
    print("=" * 78)


def cuda_time_ms(fn, warmup=5, iters=20):
    """用 CUDA Event 计时（GPU 是异步执行的，用 time.time() 会测到错误的时间）"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def exp1_device_info():
    title("实验 1: 显卡基本信息")
    p = torch.cuda.get_device_properties(0)
    print(f"   名称           : {p.name}")
    print(f"   计算能力       : sm_{p.major}{p.minor}")
    print(f"   SM 数量        : {p.multi_processor_count}")
    print(f"   显存容量       : {p.total_memory / 1024**3:.1f} GiB")
    l2 = getattr(p, "L2_cache_size", None)
    if l2:
        print(f"   L2 缓存        : {l2 / 1024**2:.0f} MiB  (测带宽时张量必须远大于它，否则测到的是 L2 带宽)")
    return l2 or 0


def exp2_bandwidth(l2_bytes):
    title("实验 2: 实测显存带宽 β")
    n_bytes = max(1 << 30, 16 * l2_bytes)   # 至少 1 GiB，远大于 L2
    x = torch.empty(n_bytes // 2, dtype=torch.bfloat16, device="cuda").uniform_()
    y = torch.empty_like(x)
    t_copy = cuda_time_ms(lambda: y.copy_(x))
    bw_copy = 2 * n_bytes / (t_copy / 1e3) / 1e9     # 读一遍 + 写一遍
    print(f"   拷贝 {n_bytes / 1024**3:.1f} GiB: {t_copy:.2f} ms → 有效带宽 {bw_copy:.0f} GB/s (读+写)")
    del x, y
    torch.cuda.empty_cache()
    return bw_copy


def exp3_peak_flops():
    title("实验 3: 实测 BF16 矩阵乘峰值算力 π")
    best = 0.0
    for n in [1024, 2048, 4096, 8192]:
        a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        t = cuda_time_ms(lambda: a @ b, iters=10)
        tflops = 2 * n ** 3 / (t / 1e3) / 1e12
        best = max(best, tflops)
        ai = 2 * n ** 3 / (2 * 3 * n * n)
        print(f"   [{n}x{n}] @ [{n}x{n}] : {t:7.3f} ms → {tflops:6.1f} TFLOP/s   (AI = {ai:6.0f})")
        del a, b
    torch.cuda.empty_cache()
    return best


def exp4_decode_linear(bw, peak):
    title("实验 4: 验证'Decode 线性层的算术强度 ≈ batch size'")
    K, N = 4096, 14336   # Llama-3-8B 的 FFN up/gate 投影形状，BF16 权重约 117 MB（远大于 L2）
    # nn.Linear 把权重存成 [out, in]，前向是 x @ W^T，这里用同样的布局
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    w_bytes = K * N * 2
    ridge = peak * 1e12 / (bw * 1e9)
    print(f"   权重 W: [{K}, {N}] BF16 = {w_bytes / 1024**2:.0f} MiB | 你这张卡的实测拐点 AI* ≈ {ridge:.0f}")
    print(f"   {'batch B':>8} | {'耗时 ms':>8} | {'理论 AI':>8} | {'实际 TFLOP/s':>12} | {'等效权重带宽 GB/s':>18}")
    times = {}
    for B in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        X = torch.randn(B, K, dtype=torch.bfloat16, device="cuda")
        t = cuda_time_ms(lambda: torch.nn.functional.linear(X, W), iters=50)
        times[B] = t
        ai = 2 * B * K * N / (2 * (B * K + K * N + B * N))
        tflops = 2 * B * K * N / (t / 1e3) / 1e12
        print(f"   {B:>8} | {t:8.3f} | {ai:8.1f} | {tflops:12.2f} | {w_bytes / (t / 1e3) / 1e9:18.0f}")
    print(f"   --> B 远小于拐点时，耗时几乎不随 B 变化（都在搬同一份权重）；B 接近拐点 {ridge:.0f} 后才开始线性增长。")
    print(f"      所以 Decode 时加大 batch 几乎是'免费'提升吞吐 —— 这就是连续批处理（Lab 12a）的物理基础。")
    slow = [B for B in times if any(times[B] > 1.3 * times[B2] for B2 in times if B2 > B)]
    if slow:
        print(f"   [注意] 注意 B = {slow} 这几档比更大的 batch 还慢：这不是测量噪声，而是 cuBLAS 的启发式规则对这些形状")
        print("      选中了效率较低的 kernel。Roofline 只给出上界，库实现不一定能贴着它走。")
        print("      推理引擎因此会对常用形状做调优、用 CUDA Graph 把 batch 补齐到固定档位、甚至手写 GEMV kernel。")
    del W
    torch.cuda.empty_cache()


def exp5_decode_attention():
    title("实验 5: 验证'Decode 注意力的算术强度 ≈ GQA 分组大小 g，与 batch 无关'")
    B, S, n_kv, d = 16, 4096, 8, 128   # 16 个请求，每个上下文 4096，8 个 KV 头
    K = torch.randn(B, n_kv, S, d, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, n_kv, S, d, dtype=torch.bfloat16, device="cuda")
    kv_bytes = 2 * K.numel() * 2
    print(f"   {B} 个请求 × 上下文 {S} × {n_kv} 个 KV 头 × d={d}: KV Cache 共 {kv_bytes / 1024**2:.0f} MiB（每步必须全部读一遍）")
    print(f"   {'每组 Q 头数 g':>12} | {'Q 头总数':>8} | {'耗时 ms':>8} | {'理论 AI':>8} | {'等效 KV 带宽 GB/s':>18}")
    for g in [1, 2, 4, 8]:
        q = torch.randn(B, n_kv, g, d, dtype=torch.bfloat16, device="cuda")   # 每个 KV 头被 g 个 Q 头共享

        def attn():
            s = (q @ K.transpose(-1, -2)) * (d ** -0.5)       # [B, n_kv, g, S]
            p = torch.softmax(s.float(), dim=-1).to(torch.bfloat16)
            return p @ V                                       # [B, n_kv, g, d]
        t = cuda_time_ms(attn)
        print(f"   {g:>12} | {g * n_kv:>8} | {t:8.3f} | {g:8d} | {kv_bytes / (t / 1e3) / 1e9:18.0f}")
    print("   --> g 从 1 到 8，FLOPs 涨了 8 倍，耗时却基本不变（瓶颈是读 KV，而 KV 字节数不变）。")
    print("      MHA 相当于 g=1，把同样的 KV 读取只用了一次；GQA/MLA 让同一份 KV 被多个 Q 头复用 → Decode 更快。")
    print("      （这里没有做算子融合，softmax 等中间结果也会读写显存，所以带宽数字偏低；FlashDecoding 会把它们融合掉。）")
    del K, V
    torch.cuda.empty_cache()


def exp6_float_pitfalls():
    title("实验 6: 浮点格式的范围与精度陷阱（第 8 篇 §6）")
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        fi = torch.finfo(dt)
        print(f"   {str(dt):<16}: 最大值 {fi.max:.3e} | 机器 ε {fi.eps:.3e}")

    print("\n   (a) 范围：70000 能表示吗？")
    print(f"       FP16: {torch.tensor(70000.0, dtype=torch.float16).item()}   ← 超过 65504，溢出成 inf")
    print(f"       BF16: {torch.tensor(70000.0, dtype=torch.bfloat16).item()}   ← 能表示，但只有约 3 位有效数字")

    print("\n   (b) 精度：256 + 1 = ?")
    print(f"       BF16: {(torch.tensor(256.0, dtype=torch.bfloat16) + 1).item()}   ← 在 256 附近 BF16 的间隔是 2，+1 被吞掉")
    print(f"       FP16: {(torch.tensor(256.0, dtype=torch.float16) + 1).item()}")

    print("\n   (c) 在 BF16 里逐个累加 1000 个 1.0:")
    s = torch.tensor(0.0, dtype=torch.bfloat16)
    one = torch.tensor(1.0, dtype=torch.bfloat16)
    for _ in range(1000):
        s = s + one
    print(f"       BF16 累加器结果 = {s.item()}   (正确答案 1000)")
    print(f"       FP32 累加器结果 = {torch.ones(1000, dtype=torch.bfloat16).float().sum().item()}")
    print("   --> 所以 Tensor Core 做 BF16 矩阵乘时内部用 FP32 累加；Softmax / RMSNorm 的求和也要先转 FP32。")


def main():
    exp6_float_pitfalls()
    if not torch.cuda.is_available():
        print("\n[注意] 没有检测到 CUDA，实验 1~5 需要 NVIDIA GPU。RTX 50 系列请安装 cu128 版本的 PyTorch（见 requirements.txt）。")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    l2 = exp1_device_info()
    bw = exp2_bandwidth(l2)
    peak = exp3_peak_flops()
    title("小结：你这张卡的 Roofline")
    print(f"   实测带宽 β ≈ {bw:.0f} GB/s | 实测 BF16 峰值 π ≈ {peak:.1f} TFLOP/s | 拐点 AI* = π/β ≈ {peak * 1e12 / (bw * 1e9):.0f} FLOP/Byte")
    print(f"   BF16 Qwen2.5-7B（约 15.2 GB 权重）batch=1 Decode 上限 ≈ {bw / 15.2:.0f} token/s")
    exp4_decode_linear(bw, peak)
    exp5_decode_attention()


if __name__ == "__main__":
    main()
