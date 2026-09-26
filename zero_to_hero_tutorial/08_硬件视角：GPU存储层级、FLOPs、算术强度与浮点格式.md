# 第 8 篇：硬件视角：GPU 存储层级、FLOPs、算术强度与浮点格式

> **阅读前须知**：第 6 篇和 Lab 08 告诉了你一个结论——"Prefill 算力受限，Decode 访存受限"。但为什么？受限到什么程度？batch 开多大才能把 GPU 喂饱？GQA 除了省显存还有没有别的好处？量化为什么能加速？这些问题都需要**把硬件放进公式里**才能回答。本篇是从"懂模型"走向"懂推理系统"的分水岭。
>
> **前置**：第 6 篇（Prefill/Decode、KV Cache）、第 3 篇（$2N$ FLOPs/token）、第 5 篇（GQA/MLA）。
>
> **配套实验**：[`lab08_memory_and_roofline/measure_gpu_roofline.py`](../lab08_memory_and_roofline/measure_gpu_roofline.py)——在你自己的 RTX 5060 Ti 上实测带宽、峰值算力、Decode 形状矩阵乘随 batch 的变化曲线，以及 BF16/FP16 的精度陷阱。

---

## 🏭 一、GPU 的基本构造：计算很便宜，搬数据很贵

### 1. 计算单元
- GPU 由几十到上百个 **SM（Streaming Multiprocessor，流式多处理器）** 组成。RTX 5060 Ti 有 36 个 SM，H100 有 132 个。
- 每个 SM 里有两类计算单元：
  - **CUDA Core**：做普通的标量浮点运算（FP32 加乘）；
  - **Tensor Core**：专门做**小矩阵乘累加**（例如一次完成 $16 \times 16$ 的块乘），低精度（BF16/FP16/FP8/FP4）吞吐是 CUDA Core 的十几倍以上。**大模型的矩阵乘几乎全部跑在 Tensor Core 上。**
- 线程以 **32 个为一组（warp）** 同步执行同一条指令。

### 2. 存储层级（越往下越大、越慢）
| 层级 | 容量量级 | 带宽量级 | 谁能访问 |
| :--- | :--- | :--- | :--- |
| 寄存器 | 每 SM 256 KB | 最快 | 单个线程 |
| 共享内存 / L1（SRAM） | 每 SM 约 100~228 KB | 全卡合计数十 TB/s | 同一个 SM 内的线程块 |
| L2 缓存 | 全卡数十 MB | 数 TB/s | 全卡 |
| **显存（HBM / GDDR）** | 16 ~ 192 GB | **0.4 ~ 8 TB/s** | 全卡 |

RTX 5060 Ti 的显存是 16GB GDDR7、128 位总线，标称带宽 **448 GB/s**。作为对比，H100 SXM 的 HBM3 带宽是 3.35 TB/s。

> 💡 **一个工厂比喻**（来自 Horace He 的 *Making Deep Learning Go Brrrr*）：Tensor Core 是一座效率极高的工厂，显存是远处的仓库，显存带宽是连接两者的公路。工厂再快，原料运不过来也只能停工。**大模型 Decode 的处境就是：工厂 90% 以上的时间在等货车。**

---

## 🔢 二、数清楚两件事：算了多少（FLOPs）、搬了多少（Bytes）

### 1. 矩阵乘的 FLOPs
$[M, K] \times [K, N]$：输出有 $MN$ 个元素，每个是 $K$ 次乘法 + $K$ 次加法：
$$ \text{FLOPs} = 2MKN $$
一个参数量为 $N_{\text{param}}$ 的模型，处理一个 token 约 $2N_{\text{param}}$ FLOPs（第 3 篇）。

### 2. 矩阵乘的访存量
每个元素 $b$ 字节（BF16 为 2），至少要读两个输入、写一个输出：
$$ \text{Bytes} = b\,(MK + KN + MN) $$

### 3. 算术强度（Arithmetic Intensity）
$$ \text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} \quad (\text{单位：FLOP/Byte，每搬 1 字节做多少次运算}) $$

---

## 📐 三、Roofline 模型：一张图判断你被谁卡住

硬件有两个上限：峰值算力 $\pi$（FLOP/s）和显存带宽 $\beta$（Byte/s）。一个算术强度为 AI 的算子，能达到的性能是：
$$ \text{可达性能} = \min(\pi, \ \text{AI} \times \beta) $$

```
 性能 (FLOP/s)
   ▲
 π ┤                 ┌──────────────────────  算力屋顶（compute-bound）
   │               ／
   │             ／
   │           ／  斜率 = 带宽 β（memory-bound）
   │         ／
   │       ／
   └──────┴───────────────────────────────►  算术强度 AI (FLOP/Byte)
          AI* = π / β  （拐点 ridge point）
```

**拐点** $\text{AI}^* = \pi / \beta$：算子的 AI 低于它，就是访存受限；高于它，就是算力受限。

| 硬件 | BF16 稠密峰值 $\pi$ | 带宽 $\beta$ | 拐点 $\text{AI}^*$ |
| :--- | :--- | :--- | :--- |
| A100 SXM | 312 TFLOP/s | 2.04 TB/s | ≈ 153 |
| H100 SXM | 989 TFLOP/s | 3.35 TB/s | ≈ 295 |
| RTX 5060 Ti | **用配套实验实测** | 448 GB/s（标称） | 实测后自己算 |

> ⚠️ 厂商宣传的 Tensor 算力常常是"稀疏"或"FP16 累加"口径，GeForce 卡在 FP32 累加（训练/推理实际用的模式）下常常只有宣传值的一半。**永远以实测为准**——这也是配套实验存在的意义。

---

## 🎯 四、核心推导：Decode 的算术强度 ≈ Batch Size

### 1. 线性层（占 Decode 权重读取的绝大部分）
Decode 时 batch 里有 $B$ 个请求，每个请求只有 1 个新 token，所以线性层是 $[B, K] \times [K, N]$，而权重 $KN$ 远大于激活 $BK$、$BN$（$B \ll K, N$）：
$$ \text{AI} = \frac{2BKN}{b(BK + KN + BN)} \approx \frac{2BKN}{b \cdot KN} = \frac{2B}{b} \overset{\text{BF16}}{=} \boxed{\ B\ } $$

**Decode 时线性层的算术强度约等于 batch size。** 这一个结论解释了一大串现象：
- batch = 1 时 AI ≈ 1，而拐点在一两百 → GPU 算力利用率只有 **1% 量级**；
- 把 batch 从 1 提到 64，每一步读的权重**几乎不变**，时间几乎不变，吞吐却提升接近 64 倍——这就是**批处理的全部意义**，也是 Lab 12a 连续批处理仿真里"每一步耗时视为常数"这个假设的来源；
- 只有当 $B$ 接近拐点（A100 上约 150）后，再加 batch 才会让每一步变慢。

### 2. Prefill
Prefill 时 $M$ = prompt 长度。例如 $M = 2000$、$K = N = 4096$，代入完整公式：
$$ \text{AI} = \frac{2MKN}{b(MK + KN + MN)} = \frac{2 \cdot 2000 \cdot 4096^2}{2(2000 \cdot 4096 + 4096^2 + 2000 \cdot 4096)} \approx 1012 $$
（注意这里 $M$ 已经不再远小于 $K, N$，上面 Decode 用的近似 $2M/b$ 只是 $M \to 0$ 时的上界，会高估一倍。）即便如此，1012 也远超拐点 → **算力受限**。
所以 Prefill 的耗时估算是：
$$ T_{\text{prefill}} \approx \frac{2 N_{\text{param}} \cdot S_{\text{prompt}}}{\pi \cdot \text{MFU}} $$
MFU（Model FLOPs Utilization）是实际达到峰值的比例，优秀实现通常 40%~70%。

### 3. Decode 中的注意力：batch 救不了它
线性层的权重被 batch 里所有请求**共享**，所以 batch 能摊薄权重读取。但**每个请求的 KV Cache 是它自己的**，只有它自己的 query 会用。
单个请求、单层、上下文长度 $S$：Q 头数 $n_h$、KV 头数 $n_{kv}$、头维度 $d_h$：
$$ \text{FLOPs} = \underbrace{2 n_h S d_h}_{QK^T} + \underbrace{2 n_h S d_h}_{PV} = 4 n_h S d_h, \qquad \text{Bytes} = \underbrace{2 S n_{kv} d_h}_{K \text{ 和 } V} \cdot b $$
$$ \text{AI}_{\text{attn}} = \frac{4 n_h S d_h}{2 S n_{kv} d_h b} = \frac{2}{b} \cdot \frac{n_h}{n_{kv}} \overset{\text{BF16}}{=} \boxed{\ \frac{n_h}{n_{kv}}\ } $$

**Decode 注意力的算术强度 = GQA 的分组大小，与 batch 无关、与上下文长度无关。**
- MHA：AI = 1，永远被带宽卡死；
- Llama-3-8B（32 Q 头 / 8 KV 头）：AI = 4；
- MLA 让大量 Q 头共享同一个压缩潜向量，AI 更高。

> 🎯 **所以 GQA/MLA 的价值有两层**：第 5 篇讲的是**省显存**（能放更多请求、更长上下文），这里推出的是**提速**（同样的 KV 字节被更多次计算复用）。长上下文时注意力的 KV 读取会超过权重读取，成为 Decode 的主要耗时——这时 GQA 分组大小直接决定速度。

### 4. 把两部分合起来：一步 Decode 的耗时模型
$$ T_{\text{step}}(B) \approx \max\left( \frac{W_{\text{bytes}} + B \cdot S \cdot \text{KV}_{\text{bytes/token}}}{\beta}, \ \frac{2 N_{\text{param}} B}{\pi} \right) $$
系统吞吐 $= B / T_{\text{step}}(B)$，单用户速度 $= 1 / T_{\text{step}}(B)$。
**加大 batch：总吞吐上升，单用户速度下降**——这就是推理服务永恒的"吞吐-延迟权衡"，Lab 08 的 `memory_calculator.py` 已经按这个公式输出了结果。

---

## 🔗 五、小算子与"算子融合"

RMSNorm、残差相加、SiLU、逐元素乘……这些算子每个元素只做几次运算，AI 远小于 1。如果每个都单独启动一个 kernel：
```
读 x (HBM) → RMSNorm → 写回 HBM → 读回来 → 乘 γ → 写回 → 读回来 → 加残差 → 写回 ...
```
大部分时间花在显存来回搬运上。**算子融合（Kernel Fusion）**把多个逐元素操作合成一个 kernel：数据只从显存读一次，在寄存器里做完所有运算，再写回一次。
- `torch.compile`、Triton、TensorRT 做的核心优化之一就是自动融合；
- **FlashAttention 本质上就是把"QKᵀ → 掩码 → Softmax → 乘 V"整个融合成一个 kernel**，而且巧妙地让 $S \times S$ 的注意力矩阵根本不落到显存上——这需要一个数学技巧（Online Softmax），见第 9 篇。

---

## 🔬 六、浮点格式：量化与混合精度的前提

### 1. 浮点数的结构
一个浮点数由三部分组成：符号位 $s$、指数位 $e$（决定**范围**）、尾数位 $m$（决定**精度**）：
$$ x = (-1)^s \times 2^{e - \text{bias}} \times (1.m)_2 $$

| 格式 | 符号/指数/尾数 | 最大值 | 相对精度（机器 ε） | 主要用途 |
| :--- | :--- | :--- | :--- | :--- |
| FP32 | 1 / 8 / 23 | $3.4 \times 10^{38}$ | $2^{-23} \approx 1.2 \times 10^{-7}$ | 累加器、优化器状态 |
| FP16 | 1 / 5 / 10 | **65504** | $2^{-10} \approx 9.8 \times 10^{-4}$ | 早期混合精度 |
| **BF16** | 1 / **8** / 7 | $3.4 \times 10^{38}$ | $2^{-7} \approx 7.8 \times 10^{-3}$ | 现代训练与推理的默认格式 |
| FP8 E4M3 | 1 / 4 / 3 | 448 | $2^{-3}$ | 前向的权重与激活 |
| FP8 E5M2 | 1 / 5 / 2 | 57344 | $2^{-2}$ | 梯度（需要更大范围） |
| FP4 E2M1 | 1 / 2 / 1 | 6 | — | Blackwell（含你的 5060 Ti）原生支持，需配合分块缩放 |

### 2. 为什么 BF16 打败了 FP16？
BF16 就是把 FP32 的尾数**直接砍掉 16 位**：指数位和 FP32 一样多，**范围与 FP32 相同**，只是精度低。
FP16 精度更高，但最大只能表示 65504，训练时激活或梯度很容易**溢出成 inf**，需要"损失缩放（loss scaling）"等额外技巧。深度学习对**范围**比对**精度**敏感得多，所以 BF16 胜出。

### 3. 低精度的陷阱：累加必须用高精度
BF16 只有 8 位有效精度（7 位尾数 + 隐含的 1）。在 256 附近，相邻两个可表示数之间的间隔是 2：
$$ 256 + 1 \overset{\text{BF16}}{=} 256 $$
如果你把 4096 个数在 BF16 里逐个累加，后面加的小数会被**全部吞掉**。所以：
- Tensor Core 做 BF16 矩阵乘时，**内部用 FP32 累加**；
- Softmax、RMSNorm 的求和通常在 FP32 下完成（Lab 07b 的实现里你会看到 `.float()`）。

配套实验会让你亲眼看到这个现象。

### 4. 整数格式与浮点格式的区别
INT8/INT4 是**均匀网格**（相邻值间隔固定为缩放因子 $S$），浮点是**对数网格**（越靠近 0 越密）。
神经网络的权重大多集中在 0 附近，所以同样 8 位，FP8 对小值更友好、INT8 对均匀分布更友好。这是第 11 篇量化的出发点。

---

## ✅ 本篇自测

1. $[1, 4096] \times [4096, 4096]$ 的 BF16 矩阵乘，FLOPs 和访存字节各是多少？AI 是多少？
2. 为什么说 Decode 时"batch 从 1 提到 32，每一步耗时几乎不变"？什么时候这个结论不再成立？
3. 推导 Decode 注意力的 AI，并说明为什么增大 batch 对它没有帮助。
4. 在一张带宽 448 GB/s 的卡上，BF16 的 Qwen2.5-7B（约 15.2 GB 权重）batch=1 Decode 的理论上限是多少 token/s？换成 INT4 权重呢？
5. BF16 和 FP16 都是 16 位，为什么训练更偏爱 BF16？
6. 为什么 Softmax 的求和要在 FP32 里做？

---

## 📚 外部资源

- Horace He：[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) ——算力/带宽/开销三分法，**必读**。
- NVIDIA：[GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) ——官方的算术强度与 Roofline 讲解。
- kipply：[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) ——用本篇的方法把推理的每一项算清楚，**必读**。
- Google DeepMind：[How to Scale Your Model](https://jax-ml.github.io/scaling-book/) ——第 1 章 Rooflines、第 4 章 Transformer Math、第 7 章 Inference，是本篇最好的进阶读物。
- [GPU MODE 讲座](https://github.com/gpu-mode/lectures) ——从 CUDA 入门到 FlashAttention 的社区课程；配合教材 *Programming Massively Parallel Processors*（PMPP，第 4 版）。
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 第 1~5 章（编程模型、存储层级）。
- Goldberg：[What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)（经典，选读）。
- Williams, Waterman, Patterson 2009：*Roofline: An Insightful Visual Performance Model for Multicore Architectures*（CACM，Roofline 的原始论文）。

---

**上一篇**：[第 7 篇：大模型的决策大脑：Logits、温度系数极限证明与采样算法](07_大模型的决策大脑：Logits、温度系数极限证明与采样算法.md) ｜ **下一篇**：[第 9 篇：FlashAttention：Online Softmax 的严格推导与 IO 复杂度](09_FlashAttention：OnlineSoftmax的严格推导与IO复杂度.md) ｜ **总路线**：[LEARNING_PATH.md](../LEARNING_PATH.md)
