"""
Lab 08: 显存占用与 Roofline 解码速率理论计算工具 (无需第三方依赖)

支持主流大模型（Llama-3、Qwen-2.5、DeepSeek 等）在不同量化精度（FP16/INT8/INT4）下：
1. 权重显存与每 Token KV Cache 显存测算
2. 指定 GPU 显存容量下的最大并发数 (Max Batch Size) 与最大上下文推演
3. 基于显存带宽 (Memory Bandwidth) 评估 Decode 阶段单 Token 理论生成耗时下界
"""

# ==========================================
# 典型模型架构参数库
# ==========================================
MODELS = {
    "Llama-3-8B": {
        "params_b": 8.03,
        "layers": 32,
        "q_heads": 32,
        "kv_heads": 8,       # GQA 组数 (8 vs 32)
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 128256,
    },
    "Llama-3-70B": {
        "params_b": 70.6,
        "layers": 80,
        "q_heads": 64,
        "kv_heads": 8,       # GQA 组数 (8 vs 64, KV 压缩 8 倍!)
        "head_dim": 128,
        "hidden_size": 8192,
        "vocab_size": 128256,
    },
    "Qwen2.5-7B": {
        "params_b": 7.61,
        "layers": 28,
        "q_heads": 28,
        "kv_heads": 4,       # GQA
        "head_dim": 128,
        "hidden_size": 3584,
        # 注意：Qwen2.5 小尺寸（0.5B/1.5B/3B）词表是 151936 且与 LM Head 共享权重，
        # 7B 及以上才是 152064 且不共享。这里两个都是 7B+ 的配置（第 5 篇 §5.2）。
        "vocab_size": 152064,
    },
    "Qwen2.5-72B": {
        "params_b": 72.7,
        "layers": 80,
        "q_heads": 64,
        "kv_heads": 8,       # GQA
        "head_dim": 128,
        "hidden_size": 8192,
        "vocab_size": 152064,
    }
}

# ==========================================
# 典型显卡硬件规格库
# 注意：bandwidth_gbs 用的是 10^9 字节/秒（厂商口径），fp16_tflops 是厂商标称的 Tensor Core 稠密峰值，
# GeForce 消费卡在 FP32 累加模式下通常只有标称值的一半，且实际可达带宽一般只有标称的 80%~90%。
# 以 lab08_memory_and_roofline/measure_gpu_roofline.py 在你机器上的实测值为准。
# ==========================================
GPUS = {
    "RTX 5060 Ti (16GB) [本机配置]": {
        "vram_gb": 16.0,
        "bandwidth_gbs": 448.0,            # 标称值（厂商标注的 448 GB/s）
        "bandwidth_gbs_measured": 380.0,   # measure_gpu_roofline.py 实测拷贝带宽
        "fp16_tflops": 48.0                # 本机实测 BF16 稠密矩阵乘约 48 TFLOP/s（FP32 累加）
    },
    "RTX 4090 (24GB)": {
        "vram_gb": 24.0,
        "bandwidth_gbs": 1008.0,
        "fp16_tflops": 165.0      # FP32 累加口径（FP16 累加口径为 330）
    },
    "NVIDIA A100-SXM4 (80GB)": {
        "vram_gb": 80.0,
        "bandwidth_gbs": 2039.0,
        "fp16_tflops": 312.0
    },
    "NVIDIA H100-SXM5 (80GB)": {
        "vram_gb": 80.0,
        "bandwidth_gbs": 3350.0,
        "fp16_tflops": 989.0
    }
}


def calculate_kv_cache_per_token_bytes(model_cfg, kv_bits=16):
    """
    计算单个 Token 产生的 Key 和 Value 向量占用的总字节数
    KV Cache bytes/token = 2 (K & V) * layers * (kv_heads * head_dim) * (kv_bits / 8)
    """
    bytes_per_elem = kv_bits / 8.0
    layers = model_cfg["layers"]
    kv_heads = model_cfg["kv_heads"]
    head_dim = model_cfg["head_dim"]
    return 2 * layers * (kv_heads * head_dim) * bytes_per_elem


def calculate_weight_memory_gb(params_b, weight_bits=16):
    """
    计算模型权重显存占用，单位 **GiB**（除以 1024^3）。

    注意这里的口径：参数量按**十进制**算（1B = 10^9 个参数），存储按**二进制**换算（1024^3）。
    所以 1B 参数在 16-bit 下是 1.86 GiB，不是"2GB"。
    这么选是为了和 GPU 的 vram_gb（16GB 卡 = 16 GiB 容量）能直接比较 ——
    上面那段 Roofline 计算用的是十进制 GB（bytes / 1e9），因为要和 GB/s 相除，两处**故意不一致**，
    下面的打印会把单位写清楚。
    """
    bytes_per_param = weight_bits / 8.0
    total_bytes = params_b * (10**9) * bytes_per_param
    return total_bytes / (1024**3)


def print_model_analysis(model_name, gpu_name, context_len=4096, batch_size=4):
    model = MODELS[model_name]
    gpu = GPUS[gpu_name]

    print("=" * 72)
    print(f"📊 模型推理性能评估: {model_name} on {gpu_name}")
    print("=" * 72)

    # 1. 权重占用
    w_fp16 = calculate_weight_memory_gb(model["params_b"], 16)
    w_int8 = calculate_weight_memory_gb(model["params_b"], 8)
    w_int4 = calculate_weight_memory_gb(model["params_b"], 4)

    print(f"📌 模型基本参数: {model['params_b']}B 参数 | {model['layers']} 层 | Q头={model['q_heads']} | KV头={model['kv_heads']}")
    print(f"📌 权重显存需求（GiB = bytes/1024³，和显卡容量同口径）:")
    print(f"   - FP16 (16-bit) : {w_fp16:6.2f} GiB")
    print(f"   - INT8 / FP8    : {w_int8:6.2f} GiB")
    print(f"   - INT4 / AWQ    : {w_int4:6.2f} GiB")

    # 2. KV Cache 计算
    kv_fp16_bytes = calculate_kv_cache_per_token_bytes(model, 16)
    kv_int8_bytes = calculate_kv_cache_per_token_bytes(model, 8)
    kv_int4_bytes = calculate_kv_cache_per_token_bytes(model, 4)

    print(f"\n📌 单 Token KV Cache 显存:")
    print(f"   - FP16 KV Cache : {kv_fp16_bytes / 1024:6.2f} KB/token")
    print(f"   - INT8 KV Cache : {kv_int8_bytes / 1024:6.2f} KB/token")
    print(f"   - INT4 KV Cache : {kv_int4_bytes / 1024:6.2f} KB/token")

    # 3. 指定场景显存拆解 (如 Batch=4, Context=4096)
    print(f"\n📌 场景评估 (并发数={batch_size}, 序列长度={context_len} tokens):")
    total_tokens = batch_size * context_len
    total_kv_fp16_gb = (total_tokens * kv_fp16_bytes) / (1024**3)
    cuda_overhead_gb = 1.2 # 运行时驱动与环境开销

    total_mem_fp16 = w_fp16 + total_kv_fp16_gb + cuda_overhead_gb
    total_mem_int4 = w_int4 + total_kv_fp16_gb + cuda_overhead_gb

    print(f"   - 全量 Token 总数   : {total_tokens} tokens")
    print(f"   - KV Cache 总占用  : {total_kv_fp16_gb:6.2f} GiB (FP16)")
    print(f"   - FP16 总显存需求  : {total_mem_fp16:6.2f} GiB (权重 {w_fp16:.1f} + KV {total_kv_fp16_gb:.1f} + 基础开销 {cuda_overhead_gb:.1f})")
    print(f"   - INT4 权重总显存  : {total_mem_int4:6.2f} GiB (权重 {w_int4:.1f} + KV {total_kv_fp16_gb:.1f} + 基础开销 {cuda_overhead_gb:.1f})")

    vram_avail = gpu["vram_gb"]
    print(f"\n🎯 硬件显存匹配状态 ({gpu_name} - {vram_avail:.0f} GiB):")
    if total_mem_fp16 <= vram_avail:
        print(f"   ✅ FP16 原生精度：显存放得下（剩余 {vram_avail - total_mem_fp16:.2f} GiB）")
    elif total_mem_int4 <= vram_avail:
        print(f"   ⚠️ FP16 显存不足 (超标 {total_mem_fp16 - vram_avail:.2f} GiB)，但 INT4/AWQ 量化后放得下 (剩余 {vram_avail - total_mem_int4:.2f} GiB)")
    else:
        print(f"   ❌ INT4 量化后仍超出容量 (超标 {total_mem_int4 - vram_avail:.2f} GiB)，需多卡张量并行 (TP) 或 Offload")
    print(f"   ⚠️ 这只是【静态显存账本】。真实部署还要算上激活、CUDA Context、碎片，"
          f"1.2 GiB 的开销常数很乐观——贴着容量上限的配置（比如 7B FP16 放在 16 GiB 上）实际常常 OOM。")

    # 4. Roofline 显存带宽解码速率下界 (Batch Size = 1)
    # 注意单位：权重按字节算，带宽是 10^9 B/s，不要把 GiB 和 GB/s 直接相除（会差 7.4%）
    # ⚠️ 这里必须用【实测】带宽，不能用标称值：标称是硬件上界，实际达不到。
    #    lab07b 的手写实现也默认用 380 GB/s，两个 lab 要同口径才能互相印证。
    bw = gpu.get("bandwidth_gbs_measured", gpu["bandwidth_gbs"])
    bw_nominal = gpu["bandwidth_gbs"]
    w_fp16_bytes = model["params_b"] * 1e9 * 2
    w_int4_bytes = model["params_b"] * 1e9 * 0.5
    # 理论单 token 最低耗时 (ms) = 模型权重读取时间
    t_fp16_ms = w_fp16_bytes / (bw * 1e9) * 1000.0
    t_int4_ms = w_int4_bytes / (bw * 1e9) * 1000.0
    speed_fp16 = 1000.0 / t_fp16_ms if t_fp16_ms > 0 else 0
    speed_int4 = 1000.0 / t_int4_ms if t_int4_ms > 0 else 0

    print(f"\n⚡ Decode 阶段理论单流最大生成速度")
    print(f"   （用实测带宽 {bw:.0f} GB/s 计算；标称 {bw_nominal:.0f} GB/s 只是硬件上界，"
          f"用它算会乐观 {(bw_nominal / bw - 1) * 100:.0f}%）")
    print(f"   - FP16 解码极限 : {t_fp16_ms:5.1f} ms/token -> 上限约 {speed_fp16:5.1f} tokens/s")
    print(f"   - INT4 解码极限 : {t_int4_ms:5.1f} ms/token -> 上限约 {speed_int4:5.1f} tokens/s (加速 {t_fp16_ms/t_int4_ms:.1f}x)")

    # 5. 更完整的 Decode 下界：每一步不仅要读一遍权重，还要读一遍【所有请求的全部 KV Cache】
    #    一步耗时 >= (权重字节 + B * S * 每token KV字节) / 带宽，这一步产出 B 个 token
    kv_bytes_all = total_tokens * kv_fp16_bytes
    t_step_ms = (w_fp16_bytes + kv_bytes_all) / (bw * 1e9) * 1000.0
    print(f"\n⚡ 场景 (B={batch_size}, S={context_len}) 下 FP16 每一步 Decode 的带宽下界（权重 + 全部 KV 各读一遍）:")
    print(f"   - 每步至少 {t_step_ms:5.1f} ms，其中读 KV 占 {kv_bytes_all / (w_fp16_bytes + kv_bytes_all) * 100:4.1f}%")
    print(f"   - 系统总吞吐上限约 {batch_size * 1000.0 / t_step_ms:6.1f} tokens/s（每个请求各 {1000.0 / t_step_ms:5.1f} tokens/s）")
    print(f"   👉 Batch 越大，同一次权重读取被越多请求分摊，总吞吐越高；但上下文越长，KV 读取越会成为新瓶颈。")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    # 自检：Llama-3-8B 每 token KV = 2 × 32 层 × 8 头 × 128 维 × 2 字节 = 128 KB（第 6 篇、第 5 篇的算例）
    assert calculate_kv_cache_per_token_bytes(MODELS["Llama-3-8B"], 16) == 131072
    # 演示本机 RTX 5060 Ti 16GB 跑 Llama-3-8B 和 Qwen2.5-7B
    print_model_analysis("Llama-3-8B", "RTX 5060 Ti (16GB) [本机配置]", context_len=2048, batch_size=2)
    print_model_analysis("Qwen2.5-7B", "RTX 5060 Ti (16GB) [本机配置]", context_len=4096, batch_size=1)
    # 演示大模型 70B
    print_model_analysis("Llama-3-70B", "NVIDIA A100-SXM4 (80GB)", context_len=4096, batch_size=4)
