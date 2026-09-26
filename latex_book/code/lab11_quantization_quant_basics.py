"""
Lab 11: 模型量化原理与数值精度仿真 (零外部依赖)

本脚本深入演示：
1. 对称 INT8 量化 (Symmetric Quantization)
2. 非对称 INT8 量化 (Asymmetric Quantization)
3. 工业级分组 INT4 量化 (Group-wise INT4，类似 AWQ/GPTQ 基础)
4. 离群值 (Outliers) 敏感性测试与误差（MSE / SNR）分析
"""

import math
import random


def mean_squared_error(orig, dequant):
    total = sum((a - b) ** 2 for a, b in zip(orig, dequant))
    return total / len(orig)


def signal_to_noise_ratio(orig, dequant):
    signal_pwr = sum(a ** 2 for a in orig) / len(orig)
    noise_pwr = sum((a - b) ** 2 for a, b in zip(orig, dequant)) / len(orig)
    if noise_pwr < 1e-12:
        return float('inf')
    return 10.0 * math.log10(signal_pwr / noise_pwr)


# =====================================================================
# 1. 对称 INT8 量化 (Zero-point = 0)
# =====================================================================
def quantize_symmetric_int8(tensor):
    """
    对称量化映射到 [-127, 127]
    Scale = max(|tensor|) / 127
    """
    max_abs = max(abs(x) for x in tensor)
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    
    # 量化到 INT8
    q_tensor = []
    for x in tensor:
        q = round(x / scale)
        q = max(-127, min(127, q))
        q_tensor.append(int(q))
        
    # 反量化
    dequant = [q * scale for q in q_tensor]
    return q_tensor, scale, dequant


# =====================================================================
# 2. 非对称 INT8 量化 (Zero-point != 0)
# =====================================================================
def quantize_asymmetric_int8(tensor):
    """
    非对称量化映射到 [0, 255]
    Scale = (max - min) / 255
    Zero_point = round(-min / scale)
    """
    t_min = min(tensor)
    t_max = max(tensor)
    if t_max == t_min:
        scale = 1.0
        zp = 0
    else:
        scale = (t_max - t_min) / 255.0
        zp = round(-t_min / scale)
        zp = max(0, min(255, zp))
        
    q_tensor = []
    for x in tensor:
        q = round(x / scale) + zp
        q = max(0, min(255, q))
        q_tensor.append(int(q))
        
    # 反量化
    dequant = [(q - zp) * scale for q in q_tensor]
    return q_tensor, scale, zp, dequant


# =====================================================================
# 3. 分组 INT4 量化 (Group-wise INT4)
# =====================================================================
def quantize_groupwise_int4(tensor, group_size=32):
    """
    将一维张量切分为长度为 group_size 的小块，每块独立进行对称 INT4 量化 [-7, 7]
    这是现代 AWQ、GPTQ 维持超高精度的核心基石。
    """
    dequant_all = []
    scales = []
    
    num_elements = len(tensor)
    for i in range(0, num_elements, group_size):
        chunk = tensor[i: i + group_size]
        max_abs = max(abs(x) for x in chunk)
        scale = (max_abs / 7.0) if max_abs > 0 else 1.0
        scales.append(scale)
        
        # 量化与反量化当前 group
        for x in chunk:
            q = round(x / scale)
            q = max(-7, min(7, q))
            dequant_all.append(q * scale)
            
    return dequant_all, scales


def run_experiment():
    print("=" * 70)
    print("[起步] 模型量化算法与误差评测 (Quantization Benchmark)")
    print("=" * 70)
    
    # 模拟一个 Transformer 层的权重分布（正态分布 + 少量极值离群点）
    random.seed(42)
    n_params = 1024
    normal_weights = [random.gauss(0, 0.05) for _ in range(n_params)]
    
    # 场景 A: 无极端离群值的标准分布
    print("\n【场景 A】: 标准均匀高斯权重 (无显著离群点)")
    print("-" * 70)
    
    # 对称 INT8
    _, _, deq_sym8 = quantize_symmetric_int8(normal_weights)
    mse_sym8 = mean_squared_error(normal_weights, deq_sym8)
    snr_sym8 = signal_to_noise_ratio(normal_weights, deq_sym8)
    
    # 非对称 INT8
    _, _, _, deq_asym8 = quantize_asymmetric_int8(normal_weights)
    mse_asym8 = mean_squared_error(normal_weights, deq_asym8)
    snr_asym8 = signal_to_noise_ratio(normal_weights, deq_asym8)
    
    # 分组 INT4 (group_size=32)
    deq_grp4, _ = quantize_groupwise_int4(normal_weights, group_size=32)
    mse_grp4 = mean_squared_error(normal_weights, deq_grp4)
    snr_grp4 = signal_to_noise_ratio(normal_weights, deq_grp4)
    
    print(f"1. 对称 INT8     : MSE = {mse_sym8:.8e} | SNR = {snr_sym8:6.2f} dB (显存压缩 50%)")
    print(f"2. 非对称 INT8   : MSE = {mse_asym8:.8e} | SNR = {snr_asym8:6.2f} dB (显存压缩 50%)")
    print(f"3. 分组 INT4(G32): MSE = {mse_grp4:.8e} | SNR = {snr_grp4:6.2f} dB (显存压缩 75%)")

    # 场景 B: 含有离群点 (Outliers) 的权重
    print("\n【场景 B】: 注入 2 个大模型中常见的显著离群点 (Outliers: 绝对值放大 50 倍)")
    print("-" * 70)
    outlier_weights = list(normal_weights)
    outlier_weights[10] = 2.5   # 正常值在 0.05 左右，此处为超大离群值
    outlier_weights[200] = -2.8
    
    # 全局对称 INT8
    _, _, deq_sym8_out = quantize_symmetric_int8(outlier_weights)
    mse_sym8_out = mean_squared_error(outlier_weights, deq_sym8_out)
    snr_sym8_out = signal_to_noise_ratio(outlier_weights, deq_sym8_out)
    
    # 全局非对称 INT8
    _, _, _, deq_asym8_out = quantize_asymmetric_int8(outlier_weights)
    mse_asym8_out = mean_squared_error(outlier_weights, deq_asym8_out)
    snr_asym8_out = signal_to_noise_ratio(outlier_weights, deq_asym8_out)
    
    # 分组 INT4
    deq_grp4_out, _ = quantize_groupwise_int4(outlier_weights, group_size=32)
    mse_grp4_out = mean_squared_error(outlier_weights, deq_grp4_out)
    snr_grp4_out = signal_to_noise_ratio(outlier_weights, deq_grp4_out)
    
    # 同为 4 bit 的公平对照：全局 INT4（整个张量只有一组 scale）
    deq_glob4_out, _ = quantize_groupwise_int4(outlier_weights, group_size=len(outlier_weights))
    snr_glob4_out = signal_to_noise_ratio(outlier_weights, deq_glob4_out)
    deq_glob4, _ = quantize_groupwise_int4(normal_weights, group_size=len(normal_weights))
    snr_glob4 = signal_to_noise_ratio(normal_weights, deq_glob4)

    print(f"1. 全局对称 INT8 (Per-Tensor) : MSE = {mse_sym8_out:.8e} | SNR = {snr_sym8_out:6.2f} dB (比场景 A 下降 {snr_sym8 - snr_sym8_out:5.2f} dB)")
    print(f"2. 全局非对称 INT8            : MSE = {mse_asym8_out:.8e} | SNR = {snr_asym8_out:6.2f} dB (比场景 A 下降 {snr_asym8 - snr_asym8_out:5.2f} dB)")
    print(f"3. 分组 INT4 (Group-wise G32) : MSE = {mse_grp4_out:.8e} | SNR = {snr_grp4_out:6.2f} dB (比场景 A 下降 {snr_grp4 - snr_grp4_out:5.2f} dB)")
    print(f"4. 对照：全局 INT4 (Per-Tensor): SNR 从 {snr_glob4:6.2f} dB 降到 {snr_glob4_out:6.2f} dB")
    # 分组量化的优势是"对离群值不敏感"，而不是"4 bit 比 8 bit 更准"
    assert snr_grp4 - snr_grp4_out < 2 and snr_sym8 - snr_sym8_out > 10, "离群值应重创全局量化、几乎不影响分组量化"
    assert snr_grp4_out > snr_glob4_out + 10, "同为 4 bit 时分组应远好于全局"

    print("\n" + "=" * 70)
    print("[思考] 工业级洞察 (Key Takeaways):")
    print("   1. 【全局量化的死穴】：当张量中存在个别极大离群值时，全局 Per-Tensor 的 Scale 被拉得极大，导致正常的小数值在量化时被无情舍入为 0，SNR 骤降。")
    print("   2. 【分组量化（Group-wise）的力量】：将权重按 Group (如 32 或 128) 切分，离群值只污染自身所在的小组，其他组的精度完好无损。")
    print("      注意比较的口径：分组 INT4 的绝对 SNR 仍低于全局 INT8（4 bit 本来就比 8 bit 粗），它的优势是【几乎不受离群值影响】，")
    print("      以及同为 4 bit 时远好于全局 INT4。")
    print("   3. 这就是为什么现代 AWQ / GPTQ 几乎一律采用 Group-wise INT4：省 75% 显存，精度损失通常很小（但要用 PPL 与下游任务实测，见第 11 篇）。")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
