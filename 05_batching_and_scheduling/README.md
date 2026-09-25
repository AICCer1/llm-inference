# 模块 05: 显存调度与连续批处理 (Continuous Batching)

## 🚧 为什么静态批处理（Static Batching）在 LLM 中失效？

在传统的计算机视觉或文本分类任务中，模型输入输出维度是固定的，**静态批处理（Static Batching）**非常有效。

但在大语言模型生成场景中，每个请求的：
1. **输入 Prompt 长度极不一致**（有的几十字，有的几千字）。
2. **生成长度（Output Tokens）难以事先预知**（有的遇到 `<eos>` 10 步就结束，有的需要生成 1000 步）。

### 静态 Batching 的两大致命缺陷：
```
Request 1: [Prompt: 10] [=== Gen: 20 ===] -> 结束! (闲置等待...) [PADDING PADDING PADDING]
Request 2: [Prompt: 30] [=================== Gen: 100 ===================] -> 最长请求!
Request 3: [Prompt: 5 ] [= Gen: 10 =] -> 结束! (闲置等待...) [PADDING PADDING PADDING PADDING]
```
1. **显存与算力气泡（Bubble）**：短请求早早结束，但由于必须作为一个 Batch 统一返回，短请求必须不断 padding 空 token 陪跑，浪费大量算力与带宽。
2. **首字延迟排队灾难**：新到达的请求必须等待当前 Batch 中**最慢的那一个请求完全结束**，才能开始下一批，导致极高的排队排班延迟。

---

## ⚡ 连续批处理 (Continuous Batching / In-flight Batching)

由 OSDI'22 论文 *Orca* 提出，并在 **vLLM** 和 **TensorRT-LLM** 中发扬光大：

> **核心思想**：调度粒度从「请求级（Request-level）」细化为「迭代步进级（Iteration-level）」。

```
Iteration t:   [Req 1: decode step 15] [Req 2: decode step 40] [Req 3: decode step 8]
Iteration t+1: [Req 1: 触发 EOS! 退出]   [Req 2: decode step 41] [Req 3: decode step 9]
                ⬇️ (槽位立即释放，新请求无缝插入！)
Iteration t+2: [Req 4: 快速 Prefill!]   [Req 2: decode step 42] [Req 3: decode step 10]
```

- **立即退场**：任何请求一旦生成结束标记（EOS）或达到最大长度，当步立刻释放槽位与 KV Cache。
- **即时插入**：等待队列中的新请求无需等待其他长请求完成，在下一步前向迭代中直接加入当前 Batch 进行 Prefill。
- **吞吐飞跃**：消除了绝大部分 Padding 气泡，GPU 吞吐量通常直接提升 **2x ~ 5x**！

---

## 📄 内存革命：PagedAttention 原理

即使有了 Continuous Batching，连续的内存分配依然会造成严重浪费：
- 在传统显存管理中，框架必须为请求预先分配一段**连续的固定最大长度显存**（如预留 4096 tokens）。
- 但大多数请求只生成了 100 个 token，导致高达 **60% ~ 80% 的内部显存碎片（Internal Fragmentation）**。

**vLLM 的解决方案**：
借鉴操作系统的**虚拟内存分页机制（Virtual Memory Paging）**：
- 将 KV Cache 切分成固定大小的物理块（Physical Blocks，例如每块 16 个 tokens）。
- 物理显存块**无需连续**，通过一个逻辑块表（Block Table）映射。
- 只有当生成了新的 16 个 token 时，才向物理池申请分配 1 个新块。
- **成效**：显存浪费率从 >60% 暴降至 **<4%**，允许同一张显卡承载成倍增加的并发请求数！

---

## 🏃 运行仿真实验

运行 [`continuous_batching_sim.py`](file:///home/yuan/antigravity/sharp-volta/llm_inference/05_batching_and_scheduling/continuous_batching_sim.py)：
```bash
python3 05_batching_and_scheduling/continuous_batching_sim.py
```
对比静态批处理与连续批处理在处理相同真实负载时的总吞吐量（Throughput）、平均等待延迟（Latency）与算力浪费（Bubble Ratio）。
