# Lab 08：显存开销与 Roofline 性能模型

在进行大模型推理系统设计或显卡选型时，有两大黄金法则：
1. **显存容量法则**：决定了你「能不能跑得下」（模型参数 + KV Cache + 运行时开销 $\le$ 显存容量）。
2. **Roofline 性能模型与访存带宽法则**：决定了你「能跑多快」（自回归解码受限于 GPU 显存带宽）。

---

## 🧮 1. 显存开销组成详解

大模型在推理运行时的显存占用主要由以下四部分组成：

$$ \text{Total Memory} = \text{Mem}_{\text{weights}} + \text{Mem}_{\text{KV\_Cache}} + \text{Mem}_{\text{activation}} + \text{Mem}_{\text{CUDA\_overhead}} $$

### (1) 模型权重 (Model Weights)
- **FP16 / BF16**: 每个参数占用 **2 字节 (Bytes)**。
  - 例如 7B 模型需要约 $7 \times 2 = 14 \text{ GB}$。
- **INT8 / FP8**: 每个参数占用 **1 字节**（7B 约 7 GB）。
- **INT4 / AWQ / GPTQ**: 每个参数占用 **0.5 字节**（7B 约 3.5 GB）。

### (2) KV Cache 显存
这是动态显存开销的最大头，随并发请求数和序列长度成正比膨胀：
对于采用分组查询注意力（GQA, Grouped-Query Attention）的模型：
- 层数 $L$
- KV 头数 $n_{\text{kv\_heads}}$，单头维度 $d_{\text{head}}$
- 并发请求数 $B$（Batch Size）
- 序列总长度 $S$（Prompt 长度 + 已生成长度）
- 存储精度字节数 $b$（FP16 为 2）

$$ \text{KV Cache (Bytes)} = 2 \times L \times (n_{\text{kv\_heads}} \times d_{\text{head}}) \times S \times B \times b $$

> 💡 **MHA vs GQA 的显存革命**：
> - 早期模型（如 LLaMA-1 65B）使用 Multi-Head Attention (MHA)，KV 头数等于 Query 头数（例如 64 头）。
> - 现代模型（如 Llama-3、Qwen2.5）几乎全部采用 GQA（例如 8 个 KV 头），KV Cache 显存直接暴降为原来的 $1/4 \sim 1/8$！

### (3) 激活值 (Activations)
在 Prefill 阶段较大（因为输入序列长），但在 Decode 阶段因为一次只输入 1 个 token，激活值极小（通常仅几十 MB 到几百 MB）。

### (4) CUDA 上下文与运行时开销
PyTorch / CUDA runtime、NCCL 通信缓冲区等，通常占用约 $1 \sim 2 \text{ GB}$。

---

## 🏎️ 2. Roofline 模型与为什么 Decode 是访存受限？

### 计算密度（Arithmetic Intensity）
$$ \text{Arithmetic Intensity (AI)} = \frac{\text{计算量 (FLOPs)}}{\text{访存数据量 (Bytes)}} $$

GPU 的极限速度受制于：
- **最大算力 (TFLOPs)**
- **显存带宽 (GB/s)**

两者的交界点为硬件的转折算术密度 $AI_{\text{roof}} = \frac{\text{Peak FLOPs}}{\text{Peak Bandwidth}}$。

| 阶段 | 输入规模 | 计算量 | 访存量 | 算术强度 $AI$ | 性能瓶颈 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prefill 阶段** | $N$ 个 token 并行 | $O(N \cdot P)$ 大矩阵乘法 | 读取一次模型权重 $P$ | 很高 ($> 100$) | **算力受限 (Compute-Bound)**，GPU 计算核心满载 |
| **Decode 阶段** | 每次仅 1 个 token | 仅对 1 个 token 做矩阵-向量乘 | 必须把全部模型权重和全部历史 KV Cache 从显存重新读一遍！ | 极低 ($\approx 1 \sim 2$) | **访存受限 (Memory-Bound)**，算力被闲置，卡在显存带宽搬运上 |

### 单 Token 解码耗时下界理论估算
在 Batch Size = 1 时，生成 1 个 token 必须把整个模型权重读取一遍：
$$ T_{\text{decode\_min}} \approx \frac{\text{模型权重体积 (Bytes)}}{\text{显存有效读取带宽 (Bytes/s)}} $$

> **例如**：在显存带宽为 448 GB/s 的显卡上运行 14 GB 的 7B FP16 模型：
> 每生成 1 个 token 至少需要 $14 \text{ GB} / 448 \text{ GB/s} \approx 31.25 \text{ ms}$（即吞吐上限约为 $32 \text{ tokens/s}$）。
> 如果通过 INT4 量化将权重压缩为 3.5 GB，理论下界降为 $3.5 / 448 \approx 7.8 \text{ ms}$（吞吐暴涨到 $128 \text{ tokens/s}$）！这就是量化在推理阶段能带来巨大速度提升的根本原因！
>
> ⚠️ 上面的 448 GB/s 是**厂商标称值**，实际达不到（本机实测约 380 GB/s，见下面"一次实测结果"）。
> 用实测值算，同一个例子的下界是 $14/380 \approx 36.8 \text{ ms}$ 而不是 31.25 ms —— **标称值会乐观约 18%**。
> `memory_calculator.py` 现在两种都打印，默认用实测值。

> 📏 **单位口径**：本节算例用**十进制 GB**（7B × 2 字节 = $14 \times 10^9$ B = 14 GB），
> 因为要和 GB/s 相除。而 `memory_calculator.py` 的权重/KV 打印用 **GiB**（除以 $1024^3$），
> 因为要和显卡容量比较。同一个 7B FP16 模型：14 GB = 13.04 GiB，**两个都对，差 7.4%**。
> 混用是这一行最常见的错误，脚本里专门有一段注释说明。

---

## 🛠️ 显存与性能计算工具使用

运行 [`memory_calculator.py`](memory_calculator.py)：
```bash
python3 lab08_memory_and_roofline/memory_calculator.py
```
可评估主流模型（Llama-3-8B/70B、Qwen-2.5-7B/72B 等）在各类 GPU（如 RTX 5060 Ti、RTX 4090、A100）下的显存与并发上限。

---

## 🔬 在你自己的显卡上实测 Roofline（需要 PyTorch）

理论推导见 [第 8 篇：硬件视角](../zero_to_hero_tutorial/08_硬件视角：GPU存储层级、FLOPs、算术强度与浮点格式.md)。运行：
```bash
.venv/bin/python lab08_memory_and_roofline/measure_gpu_roofline.py
```

它会测出你这张卡的真实带宽 β、BF16 峰值 π、拐点 AI* = π/β，并验证两个关键推导：
1. **Decode 线性层的算术强度 ≈ batch size**：batch 从 1 到 16，耗时几乎不变；
2. **Decode 注意力的算术强度 ≈ GQA 分组大小**：与 batch 无关，所以 GQA/MLA 不仅省显存，还直接提速。

本机（RTX 5060 Ti 16GB）的一次实测结果：

> ⏱️ 以下计时类数字是本机某一次运行的**量级示例**，前提是显卡空闲、已预热；GPU 降频或有其它负载时可能慢 40% 以上。请看**趋势和比例**，以你自己运行的输出为准。 例如有评审在并发负载下测到过 217~234 GB/s 的带宽。


| 指标 | 厂商标称 | 实测 |
| :--- | :--- | :--- |
| 显存带宽 | 448 GB/s | ≈ 380 GB/s（拷贝），≈ 400 GB/s（GEMV 读权重） |
| BF16 Tensor 稠密算力 | 宣传口径更高 | **≈ 48 TFLOP/s** |
| 拐点 AI* | — | **≈ 130 FLOP/Byte**：batch < 100 左右的 Decode 都是访存受限 |

> ⚠️ 实测中还会看到 batch = 32/64 的线性层比 batch = 128 还慢——这是 cuBLAS 对这些形状选中了低效 kernel，是真实存在的现象，不是噪声。它提醒你：**Roofline 是上界，实现质量决定你离上界有多远。**
