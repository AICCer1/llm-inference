# 模块 04: 模型量化技术 (INT8 / INT4 / FP8, AWQ, GPTQ)

## 🎯 什么是量化？为什么对推理如此重要？

在深度学习推理中，**量化（Quantization）是将高精度浮点数（如 FP32 / FP16）映射到低位宽定点数或微缩浮点数（如 INT8 / INT4 / FP8）的技术**。

对于大模型推理，量化带来双重巨大收益：
1. **显存容量降低**：70B 模型从 FP16 的 140GB 显存，经 INT4 压缩后仅需约 35GB，可以在单张或双张消费级显卡上加载。
2. **生成速度翻倍**：由于 Decode 阶段瓶颈在于显存带宽，将权重体积压缩至 1/4 后，显存读取时间直接缩短至 1/4，理论生成速度可提升至接近 4 倍！

---

## 📐 核心量化数学公式

### 1. 对称量化 (Symmetric Quantization)
将浮点零点映射为定点零点（即零偏置 $Z = 0$）：
- 缩放因子（Scale）：
  $$ S = \frac{\max(|X|)}{2^{b-1} - 1} $$
- 量化过程：
  $$ q = \text{clip}\left(\text{round}\left(\frac{X}{S}\right), -2^{b-1}+1, 2^{b-1}-1\right) $$
- 反量化（Dequantization）：
  $$ \hat{X} = q \times S $$

### 2. 非对称量化 (Asymmetric / Affine Quantization)
当数据分布具有明显单边偏置时，引入零点 Zero-point ($Z$)：
- 缩放因子与零点：
  $$ S = \frac{X_{\max} - X_{\min}}{2^b - 1} $$
  $$ Z = \text{round}\left(-\frac{X_{\min}}{S}\right) $$
- 量化与反量化：
  $$ q = \text{clip}\left(\text{round}\left(\frac{X}{S}\right) + Z, 0, 2^b - 1\right) $$
  $$ \hat{X} = (q - Z) \times S $$

---

## 🧱 量化粒度 (Granularity)

| 粒度模式 | 描述 | 硬件友好度 | 精度保持度 |
| :--- | :--- | :--- | :--- |
| **Per-Tensor** | 整个张量共享一个 Scale | 最高（计算开销最小） | 最差（易受离群异常值干扰） |
| **Per-Channel** | 矩阵的每个输出通道拥有独立的 Scale | 高 | 较好 |
| **Per-Group (分组量化)** | 连续每 64 或 128 个权重组成一组独立量化（如 AWQ/GPTQ 标配） | 中等 | **最好**（极大减小离群值污染范围） |

---

## 💡 工业级算法速览

1. **W4A16 (Weight-Only Quantization)**:
   - 权重存为 INT4，计算时在寄存器/SRAM 实时反量化为 FP16 与激活值做运算。
   - 代表：**AWQ (Activation-aware Weight Quantization)**（保护 1% 关键特征通道）和 **GPTQ**（二阶 Hessian 误差补偿）。
2. **W8A8 (Weight & Activation Quantization)**:
   - 权重与激活值均量化为 INT8 或 FP8，直接使用硬件 INT8/FP8 Tensor Core 进行矩阵乘法，Prefill 和 Decode 均获加速。
   - 代表：**SmoothQuant**（通过数学等价变换将激活值的离群峰值转移给权重）。
3. **FP8 量化（现代 GPU 标配）**:
   - Hopper/Ada/Blackwell 架构原生支持 E4M3 和 E5M2 两种 FP8 格式，几乎无损兼顾训练与推理。

---

## 🏃 运行量化实验

运行 [`quant_basics.py`](file:///home/yuan/antigravity/sharp-volta/llm_inference/04_quantization/quant_basics.py)：
```bash
python3 04_quantization/quant_basics.py
```
对比对称量化、非对称量化以及 Group-wise INT4 分组量化的重构误差（MSE）与信噪比（SNR）。
