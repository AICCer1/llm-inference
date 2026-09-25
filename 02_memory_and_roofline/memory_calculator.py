"""
模块 02: 显存占用与 Roofline 解码速率理论计算工具 (无需第三方依赖)

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
# ==========================================
GPUS = {
    "RTX 5060 Ti (16GB) [本机配置]": {
        "vram_gb": 16.0,
        "bandwidth_gbs": 448.0,
        "fp16_tflops": 160.0
    },
    "RTX 4090 (24GB)": {
        "vram_gb": 24.0,
        "bandwidth_gbs": 1008.0,
        "fp16_tflops": 330.0
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
    计算模型权重显存占用 (GB)
    1B 参数在 16-bit 下占用约 2GB (按照 1024^3 换算)
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
    print(f"📌 权重显存需求:")
    print(f"   - FP16 (16-bit) : {w_fp16:6.2f} GB")
    print(f"   - INT8 / FP8    : {w_int8:6.2f} GB")
    print(f"   - INT4 / AWQ    : {w_int4:6.2f} GB")

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
    print(f"   - KV Cache 总占用  : {total_kv_fp16_gb:6.2f} GB (FP16)")
    print(f"   - FP16 总显存需求  : {total_mem_fp16:6.2f} GB (权重 {w_fp16:.1f}G + KV {total_kv_fp16_gb:.1f}G + 基础开销 {cuda_overhead_gb:.1f}G)")
    print(f"   - INT4 权重总显存  : {total_mem_int4:6.2f} GB (权重 {w_int4:.1f}G + KV {total_kv_fp16_gb:.1f}G + 基础开销 {cuda_overhead_gb:.1f}G)")

    vram_avail = gpu["vram_gb"]
    print(f"\n🎯 硬件显存匹配状态 ({gpu_name} - {vram_avail}GB):")
    if total_mem_fp16 <= vram_avail:
        print(f"   ✅ FP16 原生精度：可流畅运行！(显存剩余 {vram_avail - total_mem_fp16:.2f} GB)")
    elif total_mem_int4 <= vram_avail:
        print(f"   ⚠️ FP16 显存不足 (超标 {total_mem_fp16 - vram_avail:.2f} GB)，但通过 INT4/AWQ 量化即可完美装下！(剩余 {vram_avail - total_mem_int4:.2f} GB)")
    else:
        print(f"   ❌ INT4 量化后仍超出显存容量 (超标 {total_mem_int4 - vram_avail:.2f} GB)，需使用多卡张量并行 (TP) 或更激进的 Offload！")

    # 4. Roofline 显存带宽解码速率下界 (Batch Size = 1)
    bw = gpu["bandwidth_gbs"]
    # 理论单 token 最低耗时 (ms) = 模型权重读取时间
    t_fp16_ms = (w_fp16 / bw) * 1000.0
    t_int4_ms = (w_int4 / bw) * 1000.0
    speed_fp16 = 1000.0 / t_fp16_ms if t_fp16_ms > 0 else 0
    speed_int4 = 1000.0 / t_int4_ms if t_int4_ms > 0 else 0

    print(f"\n⚡ Decode 阶段理论单流最大生成速度 (基于显存带宽 {bw} GB/s):")
    print(f"   - FP16 解码极限 : {t_fp16_ms:5.1f} ms/token -> 最大理论吞吐约 {speed_fp16:5.1f} tokens/s")
    print(f"   - INT4 解码极限 : {t_int4_ms:5.1f} ms/token -> 最大理论吞吐约 {speed_int4:5.1f} tokens/s (加速 {t_fp16_ms/t_int4_ms:.1f}x)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    # 演示本机 RTX 5060 Ti 16GB 跑 Llama-3-8B 和 Qwen2.5-7B
    print_model_analysis("Llama-3-8B", "RTX 5060 Ti (16GB) [本机配置]", context_len=2048, batch_size=2)
    print_model_analysis("Qwen2.5-7B", "RTX 5060 Ti (16GB) [本机配置]", context_len=4096, batch_size=1)
    # 演示大模型 70B
    print_model_analysis("Llama-3-70B", "NVIDIA A100-SXM4 (80GB)", context_len=4096, batch_size=4)
