# 第 9 篇：FlashAttention：Online Softmax 的严格推导与 IO 复杂度

> **阅读前须知**：FlashAttention（2022）可能是近几年影响最大的一个推理/训练优化——它**不改变任何数学结果**，却让注意力快了数倍、显存从 $O(N^2)$ 降到 $O(N)$，从而让长上下文成为可能。它的核心是一个非常漂亮的代数技巧：**Online Softmax**。本篇把它从头推导一遍。
>
> **前置**：第 2 篇（Softmax 平移不变性）、第 4 篇（注意力公式）、第 8 篇（存储层级、算术强度、算子融合）。
>
> **配套实验**：[`lab09_flash_attention/flash_attention_numpy.py`](../lab09_flash_attention/flash_attention_numpy.py)——用 NumPy 实现 Online Softmax、分块 FlashAttention 与 FlashDecoding 的 split-K 合并，逐一验证与朴素注意力完全相等；并在 GPU 上实测朴素注意力与 PyTorch SDPA 的显存差距。
> **真机版**：[`lab09_flash_attention/cuda_softmax/`](../lab09_flash_attention/cuda_softmax/)——把本节和下一节的 Online Softmax 写成**真正的 CUDA C++ kernel**，在你自己显卡上量出 DRAM 带宽、L2 命中与占用率。不需要装 CUDA toolkit。

---

## 🐢 一、朴素注意力慢在哪里？

对序列长度 $N$、头维度 $d$，朴素实现分三步，每一步都是一个独立的 kernel：
```
S = Q Kᵀ / √d      读 Q、K (2Nd)         → 写 S 到显存 (N²)
P = softmax(S)     读 S (N²)             → 写 P 到显存 (N²)
O = P V            读 P (N²)、V (Nd)     → 写 O (Nd)
```
- **显存**：必须存下 $N \times N$ 的 $S$ 和 $P$。$N = 32\text{K}$ 时单个头就是 $10^9$ 个元素，BF16 下 2GB——而一层有几十个头。
- **访存**：总共读写 $\Theta(Nd + N^2)$ 个元素。而 $d$ 通常只有 64~128，$N^2$ 项完全主导。Softmax 这一步每个元素只做几次运算，算术强度极低（第 8 篇），**GPU 大部分时间在搬运 $N^2$ 的中间矩阵**。

**自然的想法**：像第 8 篇说的那样做算子融合——把 Q、K、V 分块读进片上 SRAM，算完直接输出 O，$S$ 和 $P$ 永远不落到显存。
**障碍**：Softmax 需要**整行**的最大值和总和，才能归一化任何一个元素。只看到一个块时，你不知道这一行后面还有没有更大的值。

---

## 🧮 二、Online Softmax：一遍扫描算出 Softmax

### 1. 传统的"安全 Softmax"要扫三遍
对一行 $x_1, \dots, x_N$（第 2 篇）：
1. 第一遍求 $m = \max_i x_i$；
2. 第二遍求 $\ell = \sum_i e^{x_i - m}$；
3. 第三遍输出 $p_i = e^{x_i - m} / \ell$。

### 2. 把前两遍合成一遍
维护两个"运行中"的量：到目前为止看过的最大值 $m_j$ 和对应的和 $\ell_j$：
$$ m_j = \max(m_{j-1}, x_j), \qquad \ell_j = \ell_{j-1} \cdot e^{m_{j-1} - m_j} + e^{x_j - m_j} $$
初始 $m_0 = -\infty$，$\ell_0 = 0$。

**定理**：对所有 $j$，$\ell_j = \sum_{i=1}^{j} e^{x_i - m_j}$。

**证明（归纳法）**：
- $j = 1$：$m_1 = x_1$，$\ell_1 = 0 + e^{x_1 - x_1} = 1 = \sum_{i=1}^{1} e^{x_i - m_1}$ ✓
- 假设 $\ell_{j-1} = \sum_{i=1}^{j-1} e^{x_i - m_{j-1}}$，则
$$ \ell_j = \left(\sum_{i=1}^{j-1} e^{x_i - m_{j-1}}\right) e^{m_{j-1} - m_j} + e^{x_j - m_j} = \sum_{i=1}^{j-1} e^{x_i - m_j} + e^{x_j - m_j} = \sum_{i=1}^{j} e^{x_i - m_j} \quad \blacksquare $$

关键就是那个**修正因子** $e^{m_{j-1} - m_j}$：当最大值变大时，之前累加的和是以旧的最大值为基准的，乘上这个因子就把它们"换算"到新基准下。因为 $m_j \ge m_{j-1}$，这个因子 $\le 1$，**永远不会溢出**。

上面是逐个元素更新；换成逐块更新（一次处理一个块 $B$）形式完全一样：
$$ m_{\text{new}} = \max\left(m_{\text{old}}, \max_{i \in B} x_i\right), \qquad \ell_{\text{new}} = \ell_{\text{old}} \, e^{m_{\text{old}} - m_{\text{new}}} + \sum_{i \in B} e^{x_i - m_{\text{new}}} $$

---

## ⚡ 三、把输出也"在线"算出来：FlashAttention 的核心

注意力的一行输出是 $o = \sum_i p_i v_i = \frac{1}{\ell} \sum_i e^{x_i - m} v_i$（$x_i$ 是这一行第 $i$ 个分数 $q \cdot k_i / \sqrt{d}$）。
我们连最后的归一化都不急着做，只维护一个**未归一化的累加器**：
$$ \text{acc}_j = \text{acc}_{j-1} \cdot e^{m_{j-1} - m_j} + \sum_{i \in B_j} e^{x_i - m_j} \, v_i $$

**定理**：处理完第 $j$ 块后，$\text{acc}_j = \sum_{i \in B_1 \cup \dots \cup B_j} e^{x_i - m_j} v_i$。证明与上一节完全相同（把标量 1 换成向量 $v_i$）。
于是全部块处理完后：
$$ o = \frac{\text{acc}_{\text{final}}}{\ell_{\text{final}}} $$
**这和一次性算完整的 Softmax 再乘 V 在数学上严格相等**，不是近似。

### FlashAttention 算法（前向，单个头）
```
把 Q 按行切成若干块 Q_1..Q_Tr（每块 Br 行），K、V 切成 K_1..K_Tc（每块 Bc 行）
for 每个 Q 块 Q_i（不同的 Q 块互相独立 → 分给不同的 SM 并行）:
    把 Q_i 读进 SRAM；初始化 m = -∞, ℓ = 0, acc = 0（都在 SRAM / 寄存器里）
    for 每个 K/V 块 j:
        把 K_j、V_j 读进 SRAM
        S_ij = Q_i K_jᵀ / √d                    # Br × Bc 的小矩阵，只存在片上
        (因果掩码：若 K_j 整块都在 Q_i 之后，直接跳过；对角块内部再逐元素掩码)
        m_new = max(m, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)
        ℓ     = ℓ · exp(m - m_new) + rowsum(P_ij)
        acc   = acc · exp(m - m_new) + P_ij V_j
        m     = m_new
    O_i = acc / ℓ，写回显存；另外存下 logsumexp = m + log ℓ（训练的反向传播要用）
```
（这是 FlashAttention-2 的循环顺序：外层循环 Q，内层循环 K/V。FlashAttention-1 的外层是 K/V。）

### 它省下了什么？
- **显存**：$S$、$P$ 从不写入显存，额外空间只有每行的 $m$、$\ell$：$O(N)$。
- **访存**：Q、O 各读写一次；K、V 对每个 Q 块读一次，共 $N / B_r$ 次。设片上 SRAM 大小为 $M$（按元素计），块大小 $B_r \sim M / d$，论文证明 FlashAttention 的 HBM 访问量为 $\Theta\left(\frac{N^2 d^2}{M}\right)$，而朴素实现为 $\Theta(Nd + N^2)$。
  两者之比的量级是 $\frac{M}{d^2}$——**这是一个与 $N$ 无关的常数**。按配套实验 3 的简化模型（约 100 KB SRAM 要同时放下 Q、K、V、O 四个块），$d = 64$ 约省 6 倍，$d = 128$ 约省 1.5 倍；真实 kernel 的块形状与 L2 缓存命中会让具体数字不同，但"常数倍、与 $N$ 无关"这一点不变。
- **计算量**：FLOPs 与朴素完全相同（甚至略多一点点 exp 修正）。
- **随 $N$ 增长的真正收益是显存**：朴素实现需要 $O(N^2)$ 的额外显存，FlashAttention 只要 $O(N)$。再加上多个 kernel 融合成一个、不再产生 FP32 的 $N^2$ 中间张量，实测加速往往比访存模型预测的还大（配套实验 5 在 5060 Ti 上 $N = 4096$ 时约 16 倍）。

> 💡 **一个反直觉的教训**：FlashAttention 没有减少任何计算，只是**减少了数据搬运**，就换来了数倍加速。这正是第 8 篇 Roofline 的核心观点——在访存受限的区域，优化的对象是字节而不是 FLOPs。它的反向传播更进一步：**宁可重新计算 $S$ 和 $P$，也不去读存下来的 $N^2$ 矩阵**，因为重算比读显存更便宜。

**想亲手验证这句话？** [`lab09_flash_attention/cuda_softmax/`](../lab09_flash_attention/cuda_softmax/) 把本节的 Online Softmax 写成真正的 CUDA C++ kernel，在你自己显卡上量出：三个实现**都贴着 380 GB/s 的 Roofline**（也就是都"优化到位了"），差别只在搬运了 268 MB 还是 134 MB —— 约 1.9x。里面还有一个反直觉的结果：把整行缓存进 shared memory 的版本在 $N = 8192$ 时**反而掉到 74%**，因为 shared 占用把 SM 的常驻 block 数从 6 压到 2。**这正是上面说的"分块"为什么必须存在。**

---

## 🧩 四、FlashDecoding：把同一个技巧用在 Decode 上

### 1. 问题
Decode 时每个请求的 Q 只有 **1 行**。FlashAttention 靠"不同的 Q 块分给不同的 SM"来并行，现在只剩 batch × 头数 这么多个独立任务。batch = 1、32 个头、上下文 64K 时，只有 32 个任务，而 H100 有 132 个 SM——大部分 SM 在闲着，每个任务却要顺序扫完 64K 个 key。

### 2. 解法：沿 K/V 的序列维度切分（split-K）
把长度为 $S$ 的 KV Cache 切成 $s$ 段，**每段独立**算出自己的 $(m^{(r)}, \ell^{(r)}, \text{acc}^{(r)})$，分给不同的 SM 并行。最后用 Online Softmax 的规则**合并**：
$$ m = \max_r m^{(r)}, \qquad \ell = \sum_r \ell^{(r)} e^{m^{(r)} - m}, \qquad \text{acc} = \sum_r \text{acc}^{(r)} e^{m^{(r)} - m}, \qquad o = \text{acc} / \ell $$

**为什么能这样合并？** 因为第三节的定理说明 $\text{acc}^{(r)} = \sum_{i \in \text{段}r} e^{x_i - m^{(r)}} v_i$，乘以 $e^{m^{(r)} - m}$ 后正好变成 $\sum_{i \in \text{段}r} e^{x_i - m} v_i$，所有段相加就是完整的和。
这个合并操作**满足结合律**，所以切成几段、按什么顺序合并都不影响结果——这也是为什么它能用在分布式场景（例如把超长上下文的 KV 分到多张卡上，Ring Attention）。

### 3. 与 PagedAttention 的关系
PagedAttention（第 12 篇）把 KV Cache 分成固定大小的物理块，块在显存里**不连续**。它的 kernel 本质上就是"一块一块读 KV、用 Online Softmax 累加"——**Online Softmax 让注意力天然可以分块计算，这是分页 KV Cache 能高效工作的数学前提**。

---

## 🗺️ 五、版本演进一览

| 版本 | 年份 | 关键改进 |
| :--- | :--- | :--- |
| Online Softmax（Milakov & Gimelshein） | 2018 | 一遍扫描求 Softmax 的归一化因子 |
| FlashAttention | 2022 | 分块 + Online Softmax + 反向重计算，IO 感知 |
| FlashAttention-2 | 2023 | 交换内外循环、减少非矩阵乘运算、更好的并行划分，约 2 倍提速 |
| FlashDecoding | 2023 | Decode 时沿 KV 序列切分并行 |
| FlashAttention-3 | 2024 | 针对 Hopper：异步数据搬运（TMA）、warp 专门化、FP8 |

在 PyTorch 里，`F.scaled_dot_product_attention` 会自动选择 FlashAttention 等融合 kernel；vLLM、SGLang 则使用 FlashAttention / FlashInfer 等专门的推理注意力库。

---

## ✅ 本篇自测

1. 朴素注意力的显存和 HBM 访存量各是多少量级？为什么 $d$ 很小时问题特别严重？
2. 写出 Online Softmax 的更新公式，并用归纳法证明 $\ell_j = \sum_{i \le j} e^{x_i - m_j}$。
3. 修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 为什么永远不会溢出？
4. FlashAttention 的 FLOPs 比朴素实现少吗？那它为什么快？
5. FlashDecoding 解决的是什么场景下的什么问题？写出两段部分结果的合并公式。
6. 为什么说 Online Softmax 是 PagedAttention 的数学前提？

---

## 📚 外部资源

- Zihao Ye：[From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)（华盛顿大学 CSE 599M 讲义，推导最清晰的短文，**必读**）
- Milakov & Gimelshein 2018：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- Dao et al. 2022：[FlashAttention](https://arxiv.org/abs/2205.14135)；Dao 2023：[FlashAttention-2](https://arxiv.org/abs/2307.08691)；Shah et al. 2024：[FlashAttention-3](https://arxiv.org/abs/2407.08608)
- Stanford CRFM 博客：[Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- Triton 官方教程：[Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)（用 Python 写出真正在 GPU 上跑的 FlashAttention；先做完前面的矩阵乘教程）
- 代码：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)；[GPU MODE 讲座](https://github.com/gpu-mode/lectures) 中关于 FlashAttention 的几讲

---

**上一篇**：[第 8 篇：硬件视角：GPU 存储层级、FLOPs、算术强度与浮点格式](08_硬件视角：GPU存储层级、FLOPs、算术强度与浮点格式.md) ｜ **下一篇**：[第 10 篇：投机解码：无偏性证明与期望加速比](10_投机解码：无偏性证明与期望加速比.md) ｜ **总路线**：[LEARNING_PATH.md](../LEARNING_PATH.md)
