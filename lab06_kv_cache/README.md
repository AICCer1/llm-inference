# Lab 06：KV Cache 原理与自回归解码基准测试

## 💡 为什么需要 KV Cache？

在大语言模型（如 GPT、Llama、Qwen 等 Decoder-only 架构）生成文本时，是**逐字（Token-by-Token）生成**的：

```
Prompt: "人工智能的未来是"
Step 1 -> "星"
Step 2 -> "辰"
Step 3 -> "大"
Step 4 -> "海"
```

在标准的 Self-Attention 计算中：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 朴素实现的问题（无 KV Cache）：
生成第 $t$ 个 token 时，把前 $t-1$ 个 token 连同最新 token 一起重新送入模型：
- 重新计算前 $t-1$ 个 token 的 $Q, K, V$ 投影矩阵。
- 但事实上，**历史 token 的 Key 和 Value 向量是固定不变的**！
- 每次生成都会重复计算历史所有 token 的 $K$ 和 $V$，导致总计算复杂度高达 $O(N^2)$，生成越到后面越慢。

### KV Cache 的优化思路：
- 在 **Prefill 阶段**（首字），计算出 Prompt 中所有 token 的 $K$ 和 $V$，并保存在显存中（KV Cache）。
- 在 **Decode 阶段**（后续生成），每次**只输入最新的 1 个 token**：
  - 只对最新 token 计算 $q_{\text{new}}, k_{\text{new}}, v_{\text{new}}$。
  - 把 $k_{\text{new}}, v_{\text{new}}$ 拼接到缓存列表中：$K_{\text{all}} = [K_{\text{cache}}, k_{\text{new}}]$。
  - 用单个查询向量 $q_{\text{new}}$ 与整个 $K_{\text{all}}$ 做点积得到注意力分数，加权聚合 $V_{\text{all}}$。
- 这样，单步生成的计算复杂度从 $O(t)$ 矩阵乘降为向量-矩阵乘，避免了大量冗余算力消耗！

---

## 🏃 运行测试

本目录下提供了 [`kv_cache_benchmark.py`](kv_cache_benchmark.py)，演示：
1. 朴素生成 vs KV Cache 生成的数学等价性（验证输出完全一致）。
2. 每步生成耗时随序列长度增加的变化趋势对比。

```bash
python3 lab06_kv_cache/kv_cache_benchmark.py
```
*(本脚本自适应纯 Python 或 NumPy，即使暂未安装任何第三方库也能直接运行！)*

---

## 🔗 继续深入

本模块用的是随机权重的单层注意力。想在**真正训练过的模型**上验证同样的结论（输出逐 token 一致、缓存字节数与公式逐字节相等、每步耗时对比），请做：
- [Lab 05](../lab05_train_tiny_llama/)：自己训练的迷你 Llama（`generate.py`）；
- [Lab 07b](../lab07b_qwen_from_scratch/)：真实的 Qwen2.5-0.5B，上下文越长 KV Cache 的优势越大（2048 时约快一个数量级）。

注意：KV Cache 消除的是对历史 token 重复做投影 / FFN 的冗余，**注意力本身每步仍要读完整个缓存**，长上下文时这会成为新的瓶颈（第 6 篇 §3 的补充说明、第 8 篇 §4.3）。
