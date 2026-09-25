# 模块 02: 显存开销与 Roofline 性能模型

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

---

## 🛠️ 显存与性能计算工具使用

运行 [`memory_calculator.py`](file:///home/yuan/antigravity/sharp-volta/llm_inference/02_memory_and_roofline/memory_calculator.py)：
```bash
python3 02_memory_and_roofline/memory_calculator.py
```
可评估主流模型（Llama-3-8B/70B、Qwen-2.5-7B/72B 等）在各类 GPU（如 RTX 5060 Ti、RTX 4090、A100）下的显存与并发上限。
