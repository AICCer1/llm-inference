# 毕业设计：工业级推理引擎实践（vLLM / llama.cpp）

> 这是整条路线的**毕业设计**。前面的 Lab 03~13 里你已经亲手实现了推理引擎的每一个核心部件；现在去使用真正的引擎，并且**每一个观测到的数字，都要能用前面的公式解释**。
>
> 前置：第 8、10、11、12 篇；Lab 08（实测带宽）、Lab 07b（手写 Qwen）、Lab 12b（PagedAttention）。

## 原则：先预测，再测量，再解释

每做一个实验，先用 `lab08_memory_and_roofline/memory_calculator.py` 和第 8 篇的公式**写下你的预测**，再运行，最后解释差距。这比跑出一个漂亮的数字重要得多。

---

## 🧪 任务 1：用 vLLM 部署，并测出吞吐-延迟曲线

**安装**：建议单独建一个虚拟环境（vLLM 会固定自己需要的 PyTorch 版本，可能和本仓库的 `.venv` 冲突）。RTX 50 系显卡需要支持 CUDA 12.8 的较新版本，按 [vLLM 官方安装文档](https://docs.vllm.ai/) 操作。WSL2 下可以正常使用。

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.85
```

**先预测**：
- Qwen2.5-1.5B：28 层、2 个 KV 头、head_dim 128 → 每 token KV = $2 \times 28 \times 2 \times 128 \times 2 = 28{,}672$ 字节 ≈ 28 KB；
- 权重 BF16 约 3.1 GB，按你实测的带宽（本机空闲热态约 380 GB/s，负载下会更低），单流 Decode 上限 ≈ 120 token/s；
- 16GB × 0.85 扣掉权重和激活后，大约还剩多少 GB 给 KV？能放多少 token？

**再测量**：
1. 看 vLLM 启动日志里的 KV Cache 容量（不同版本措辞不同，通常会打印 KV Cache 的 token 数或 GPU block 数与最大并发）。和你的预测一致吗？block 数 × 16 = token 数？
2. 压测：
   ```bash
   python3 capstone_engines_practice/bench_openai_server.py --concurrency 1 4 16 64 --max-tokens 256
   ```
3. 画出（或列出）"并发 → 总吞吐、TPOT P50"的关系。

**要能解释**：
- 并发 1 的单流速度为什么低于 120 token/s，但比 Lab 07b 的手写实现（约 20ms/token）快得多？（提示：CUDA Graph、融合 kernel，第 12 篇 §7）
- 并发从 1 到 16，总吞吐为什么几乎线性增长、TPOT 却几乎不变？到多少并发开始 TPOT 明显变差？（第 8 篇 §4：拐点约 130）

---

## 🧪 任务 2：前缀缓存

```bash
python3 capstone_engines_practice/bench_openai_server.py --shared-prefix-tokens 3000 --concurrency 4 --max-tokens 64
```
分别在开启和关闭前缀缓存时运行（新版 vLLM 默认开启；关闭参数通常是 `--no-enable-prefix-caching`，以你的版本 `vllm serve --help` 为准）。

**预测**：3000 token 的 Prefill 按第 8 篇 $2N \cdot S / (\pi \cdot \text{MFU})$ 估算要多少毫秒？命中缓存后 TTFT 应该降到多少？

---

## 🧪 任务 3：长 Prefill 对 Decode 的干扰（Chunked Prefill）

同时跑两个压测：一个并发 16、短问题、长输出（模拟正在聊天的用户）；另一个偶尔发送 3000+ token 的长提示词。观察第一个的 **TPOT P99** 是否出现尖刺。
调整 `--max-num-batched-tokens`（每步 token 预算）重复实验，用 Lab 12b 实验 5 的模型解释你看到的现象。

---

## 🧪 任务 4：llama.cpp + GGUF 量化

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
# 从 Hugging Face 上 Qwen 官方的 GGUF 仓库下载 Qwen2.5-7B-Instruct 的 Q4_K_M 版本
./build/bin/llama-bench -m <你的 gguf 文件> -ngl 99
```
`llama-bench` 会报告 `pp512`（Prefill 512 token 的吞吐）和 `tg128`（Decode 的吞吐）。

**预测**：
- Q4_K_M 平均约 4.8 bit/参数，7.6B 参数约 4.6 GB → `tg` 上限 ≈ 380 / 4.6 ≈ 80 token/s；
- `pp` 是算力受限：约 $\frac{48 \times 10^{12} \times \text{MFU}}{2 \times 7.6 \times 10^9}$ token/s。
- BF16 的 7B（15.2 GB）在 16GB 卡上几乎放不下——这就是你这张卡上量化的第一价值（第 11 篇）。

**要能解释**：为什么量化让 `tg` 提升了约 3 倍，而 `pp` 提升远没有那么多，甚至可能变慢？（提示：W4A16 在计算前要反量化，Prefill 是算力受限的）

---

## 🧪 任务 5：投机解码

在 vLLM（配置草稿模型或 n-gram 投机）或 llama.cpp（`llama-speculative`，目标 Qwen2.5-7B + 草稿 Qwen2.5-0.5B）上开启投机解码，测单流速度与接受率。
用第 10 篇的公式 $\frac{1 - \alpha^{K+1}}{(1 - \alpha)(Kc + 1)}$ 代入实测接受率和两模型的耗时比，预测加速比并与实测对比。再把并发调到 32，投机解码还有收益吗？为什么？

---

## 📖 任务 6：源码阅读路线

读源码的顺序很重要——从小到大：

1. **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**（约 1200 行 Python，实现了 vLLM 的核心：调度、块管理、前缀缓存、CUDA Graph、张量并行）。对照表：
   | nano-vllm 里的部分 | 本仓库对应 |
   | :--- | :--- |
   | 调度器（Prefill / Decode、抢占） | Lab 12a、第 12 篇 §2、§3.4 |
   | 块管理器（引用计数、哈希前缀缓存） | Lab 12b 的 `BlockAllocator`、实验 4 |
   | 模型实现（Qwen） | Lab 07b 的 `QwenFromScratch` |
   | 注意力（调用 FlashAttention 的变长 / 分页接口） | Lab 09、第 9 篇 |
   | CUDA Graph 捕获 | Lab 07b 性能分析、练习 5 |
   | 张量并行的线性层 | Lab 13、第 13 篇 |
2. **[gpt-fast](https://github.com/pytorch-labs/gpt-fast)**（不到 1000 行，用 `torch.compile` + 量化 + 投机解码把单流速度逼近带宽极限）。
3. **vLLM 本体**：先读官方文档的架构设计部分，再按"入口 → 引擎核心 → 调度器 → KV Cache 管理 → 模型执行器 → 注意力后端"的顺序读。
4. **llama.cpp**：从 `ggml` 的计算图与量化格式（`Q4_K` 等）入手，看它如何在 CPU / GPU 上做融合的反量化矩阵乘。

---

## ✅ 毕业标准

完成下面这件事，你就真正走完了这条路线：

> 选定一个模型和你的 5060 Ti，写一份一页纸的报告：**预测**它在 vLLM 上的单流速度、最大并发、TTFT（带/不带前缀缓存）、W4 量化后的速度，给出每个预测的公式；然后**实测**，逐项解释预测与实测之间的差距来自哪里。

如果每一个差距你都能说清楚是哪一篇、哪一个公式、哪一个工程因素造成的，你就不再是"库的使用者"了。
