# 第 12 篇：推理服务系统：PagedAttention、前缀缓存与调度

> **阅读前须知**：前面九篇都在讲"**一个请求**怎么算得又对又快"。真实的推理服务同时面对成百上千个请求：它们长短不一、随时到达、随时结束，还常常共享同样的系统提示词。本篇讲推理引擎（vLLM、SGLang、TensorRT-LLM）在**系统层面**解决的问题：怎么衡量好坏、显存怎么管、请求怎么排。
>
> **前置**：第 6 篇（KV Cache 显存公式）、Lab 12a（连续批处理仿真）、第 8 篇（Decode 耗时模型 $T_{\text{step}}(B)$）、第 9 篇（Online Softmax 可分块计算）。
>
> **配套实验**：[`lab12b_paged_attention/paged_kv_cache_sim.py`](../lab12b_paged_attention/paged_kv_cache_sim.py)——实现块分配器、块表、分页注意力、写时复制、前缀缓存与 Chunked Prefill 的仿真。

---

## 📏 一、先定义"好"：推理服务的指标体系

| 指标 | 含义 | 用户感受 | 主要受谁影响 |
| :--- | :--- | :--- | :--- |
| **TTFT**（Time To First Token） | 从请求到达，到第一个 token 返回 | "它开始回答了吗" | 排队时间 + Prefill 时间 |
| **TPOT / ITL**（Time Per Output Token / Inter-Token Latency） | 相邻两个输出 token 的间隔 | "打字速度够不够流畅" | 每一步 Decode 的耗时 |
| **端到端延迟** | $\text{TTFT} + \text{TPOT} \times (n_{\text{out}} - 1)$ | 总等待时间 | 两者之和 |
| **吞吐（Throughput）** | 每秒生成的总 token 数 / 完成的请求数 | 老板关心的成本 | batch 大小、显存利用率 |
| **Goodput** | 满足延迟目标（SLO，如 TTFT < 1s 且 TPOT < 50ms）的请求吞吐 | 真正有意义的吞吐 | 以上全部 |

工业界通常看 **P50 / P99 分位数**，而不是平均值——一次偶发的 2 秒卡顿会被用户记住。

### Little 定律：并发数从哪来
排队论的 Little 定律：**系统中平均请求数 = 到达率 × 平均停留时间**，$L = \lambda W$。
例：每秒到达 10 个请求，每个请求平均要生成 10 秒 → 系统里同时约有 **100 个**请求在跑。每个请求上下文 2K token、每 token KV 128 KB（Llama-3-8B），光 KV Cache 就要 $100 \times 2048 \times 128\text{KB} \approx 25$ GB。
**并发数由业务决定，显存必须装得下它**——这就是为什么 KV Cache 的显存管理是推理系统的核心问题。

---

## 🔁 二、迭代级调度回顾：为什么连续批处理成立

Lab 12a 已经仿真过：静态批处理要等整批最长的请求结束，连续批处理在**每一步**都可以让结束的请求离开、新请求加入（Orca, 2022）。
它能成立的物理基础是第 8 篇的推导：**Decode 线性层的算术强度 ≈ batch size**，在 batch 远小于拐点时，一步的耗时几乎与 batch 无关。所以"多塞一个请求进这一步"几乎是免费的。

但连续批处理立刻带来两个新问题：
1. 请求随时进出，**KV Cache 的显存怎么分配和回收？**（第三~五节）
2. 新请求的 Prefill 很重，插进正在 Decode 的批次里会怎样？（第六节）

---

## 🧱 三、KV Cache 的显存碎片：PagedAttention 要解决的问题

### 1. 传统做法：按最大长度预留连续显存
请求到达时，并不知道它最终会生成多少 token，于是只能按 `max_len`（例如 2048）预留一整块**连续**显存：
```
请求 A (实际用 300)  [■■■■■■□□□□□□□□□□□□□□□□□□□□□□□□□□□□]  ← 预留 2048，只用了 300
请求 B (实际用 1200) [■■■■■■■■■■■■■■■■■■■■■□□□□□□□□□□□□□□]
```
- **内部碎片**：预留了却没用上的部分；
- **外部碎片**：不同大小的连续块反复分配释放后，空闲显存被切成零碎的小段，总量够但凑不出一整块。

vLLM 论文测量发现，当时的系统里 KV Cache 显存**只有 20%~38% 真正存了 token**。

### 2. PagedAttention：把操作系统的虚拟内存搬过来
操作系统早在几十年前就解决过同样的问题——**分页**：
| 操作系统 | PagedAttention |
| :--- | :--- |
| 进程 | 请求（序列） |
| 虚拟页 | 逻辑块（第 0 块 = 第 0~15 个 token） |
| 物理页框 | 显存里的物理块（每块存 16 个 token 的 K、V） |
| 页表 | **块表（Block Table）**：逻辑块号 → 物理块号 |
| 缺页时分配 | 写满一块时才分配下一块 |

```
请求 A 的块表: [逻辑0 → 物理7, 逻辑1 → 物理2, 逻辑2 → 物理9]   ← 物理块不需要连续
请求 B 的块表: [逻辑0 → 物理4, 逻辑1 → 物理0, ...]
```
- 每个请求最多只浪费**最后一个块里没写满的部分**（< 16 个 token），浪费率 < 4%；
- 没有外部碎片：所有块一样大，任何空闲块都能用；
- 结果：同样的显存能容纳 **2~4 倍** 的并发请求，吞吐相应提升。

### 3. 代价：注意力 kernel 要"查表读 KV"
KV 不再连续，注意力计算时要按块表逐块读取——而第 9 篇已经证明，Online Softmax 让注意力可以**逐块累加**，所以分页几乎不影响计算效率。这是 PagedAttention 在数学上可行的原因。

### 4. 显存不够时怎么办：抢占
请求在运行中不断变长，块可能被分完。此时调度器**抢占**一部分请求：
- **换出（Swap）**：把它的 KV 块拷到 CPU 内存，之后再换回；
- **重算（Recompute）**：直接丢掉它的 KV，之后重新 Prefill（Prefill 是算力受限的，对短序列往往比经 PCIe 拷贝更快）。

---

## 👯 四、共享：引用计数与写时复制

同一个提示词需要生成多个回答时（并行采样 $n > 1$、束搜索），它们的提示词 KV **完全相同**。有了块表，共享变得非常自然：
- 多个序列的块表指向**同一组物理块**，每个物理块维护一个**引用计数**；
- 当某个序列要往一个**被共享且未写满**的块里追加 token 时，先**复制一份**给自己再写（**写时复制，Copy-on-Write**）——这同样是操作系统 `fork()` 的经典技术。

并行采样 4 个回答时，提示词部分的 KV 只存 1 份而不是 4 份。

---

## 🗂️ 五、前缀缓存：跨请求复用 KV

很多请求的**开头**是一样的：
- 所有请求共享同一段几千 token 的**系统提示词**；
- 多轮对话中，第 $k$ 轮的输入 = 前 $k-1$ 轮的全部内容 + 新问题；
- Few-shot 示例、RAG 中被反复检索到的同一篇文档、Agent 的工具描述。

这些前缀的 KV 完全相同（同样的 token、同样的位置 → 同样的 K、V，RoPE 也一样）。**前缀缓存**让请求结束后 KV 块不立即释放，而是留着等下一个相同前缀的请求复用：
- **vLLM 的自动前缀缓存**：对每个**写满的块**，用"它自己的 token + 它之前所有 token"算一个哈希值作为键；新请求逐块查哈希，命中就直接复用物理块；
- **SGLang 的 RadixAttention**：用**基数树（Radix Tree）**组织所有缓存的前缀，按 LRU 淘汰。

**收益**：命中的部分**完全跳过 Prefill**。系统提示词 4000 token、用户问题 50 token 时，TTFT 可能降低一个数量级。

> ⚠️ 前缀必须**逐 token 完全相同**且从第一个 token 开始。把会变化的内容（例如当前时间）放在系统提示词开头，会让整个缓存失效——这是写提示词时要注意的工程细节。

---

## ⚖️ 六、Prefill 与 Decode 的冲突：Chunked Prefill 与 P/D 分离

### 1. 问题：一个长 Prefill 会让所有人卡住
连续批处理把新请求的 Prefill 和其他请求的 Decode 放在同一步里。第 8 篇告诉我们：
- Decode 一步耗时 ≈ 读一遍权重的时间（例如 10ms）；
- 一个 8000 token 的 Prefill 是算力受限的，可能要几百毫秒。

它们在同一步里，**这一步的耗时就由 Prefill 决定**，正在 Decode 的几十个用户会同时看到一次几百毫秒的"卡顿"（TPOT 尖刺）。

### 2. Chunked Prefill（Sarathi-Serve, 2024）
把长 Prefill **切成若干块**（例如每块 512 token），每一步只做一块，和其他请求的 Decode 拼在一起。每一步设一个 **token 预算**：
- 每一步的耗时有了上界，Decode 用户不再卡顿；
- 而且 Decode 本来算力闲置（访存受限），顺便塞进一段 Prefill，正好把闲置的算力用上——**两种瓶颈互补**。

### 3. P/D 分离（DistServe、Mooncake，2024）
更彻底的做法：Prefill 和 Decode **放到不同的 GPU 上**，Prefill 算完后把 KV Cache 通过高速网络传给 Decode 节点。
- 理由正是第 8 篇：两者的瓶颈完全相反（算力 vs 带宽），硬件配置和并行策略可以分别优化；
- 代价：KV Cache 需要跨机传输，对网络带宽要求高。

---

## 🧰 七、一个现代推理引擎里还有什么

| 技术 | 解决的问题 | 本仓库对应 |
| :--- | :--- | :--- |
| CUDA Graph | Decode 一步有几百个小 kernel，启动开销大（Lab 07b 的 13%） | Lab 07b 练习 5 |
| 融合的注意力 kernel（FlashAttention / FlashInfer） | 第 9 篇 | Lab 09 |
| 量化（权重 / KV Cache） | 显存与带宽 | Lab 11、第 11 篇 |
| 投机解码 | Decode 串行、算力闲置 | Lab 10、第 10 篇 |
| 张量并行 / 专家并行 | 单卡放不下 | 第 13 篇、Lab 13 |
| 结构化输出（JSON Schema 约束解码） | 让输出满足格式 | 第 7 篇的延伸：对 logits 做掩码 |
| 多 LoRA 服务 | 一个基座模型服务多个微调版本 | 选读 S-LoRA |

---

## ✅ 本篇自测

1. TTFT 和 TPOT 分别由什么决定？为什么服务要看 P99 而不只看平均？
2. 用 Little 定律估算：每秒 5 个请求、每个请求平均耗时 20 秒、平均上下文 4K、每 token KV 128 KB，需要多少 KV Cache 显存？
3. 预留连续显存为什么会浪费？PagedAttention 每个请求最多浪费多少？
4. 为什么说 Online Softmax（第 9 篇）是 PagedAttention 的数学前提？
5. 写时复制在什么时候触发？
6. 为什么把"当前时间"写在系统提示词开头会破坏前缀缓存？
7. 一个长 Prefill 为什么会让所有正在 Decode 的请求卡顿？Chunked Prefill 如何解决？为什么说它让"两种瓶颈互补"？

---

## 📚 外部资源

- Kwon et al. 2023：[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（vLLM 论文，**必读**），以及 [vLLM 博客](https://blog.vllm.ai/2023/06/20/vllm.html)
- Yu et al. 2022：[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)（OSDI'22，迭代级调度）
- Anyscale 博客：[How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- Zheng et al. 2023：[SGLang (RadixAttention)](https://arxiv.org/abs/2312.07104)
- Agrawal et al. 2023/2024：[Sarathi](https://arxiv.org/abs/2308.16369)、[Sarathi-Serve (Chunked Prefill)](https://arxiv.org/abs/2403.02310)
- Zhong et al. 2024：[DistServe (P/D 分离)](https://arxiv.org/abs/2401.09670)；Qin et al. 2024：[Mooncake](https://arxiv.org/abs/2407.00079)
- vLLM 官方文档：[docs.vllm.ai](https://docs.vllm.ai/) 中的 Design / Architecture 部分
- Lilian Weng：[Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)（推理优化综述）

---

**上一篇**：[第 11 篇：量化：从每位 6 dB 到 SmoothQuant、AWQ 与 GPTQ](11_量化：从每位6dB到SmoothQuant、AWQ与GPTQ.md) ｜ **下一篇**：[第 13 篇：多卡并行推理：张量并行、流水线并行与专家并行](13_多卡并行推理：张量并行、流水线并行与专家并行.md) ｜ **总路线**：[LEARNING_PATH.md](../LEARNING_PATH.md)
