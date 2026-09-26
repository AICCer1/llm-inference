# Lab 12b：PagedAttention、写时复制、前缀缓存与 Chunked Prefill

> 对应理论：[第 12 篇：推理服务系统](../zero_to_hero_tutorial/12_推理服务系统：PagedAttention、前缀缓存与调度.md)
> 前置：第 6 篇、Lab 12a、第 8 篇、第 9 篇（Online Softmax）。
> 依赖：NumPy。

## 运行

```bash
.venv/bin/python lab12b_paged_attention/paged_kv_cache_sim.py
```

| 实验 | 你会看到 |
| :--- | :--- |
| 1. 分页注意力 | 三个请求交替写入，物理块号杂乱（如 `[49, 26, 32, 50]`），逐块 Online Softmax 结果与连续存储误差 ~1e-16 |
| 2. 显存利用率 | 同样 64K token 的显存：预留 `max_len` 只能服务 32 个请求（浪费 78%），分页可服务 144 个（浪费 1.7%） |
| 3. 写时复制 | 并行采样 n=4：共享提示词块只存一份，只有"共享且未写满"的最后一块被复制，而且只复制 n−1 次（最后一个子序列可原地写入），省 50% |
| 4. 前缀缓存 | 1000 token 系统提示词：84% 的 token 免去 Prefill；改掉第 1 个 token 后缓存全部失效 |
| 5. Chunked Prefill | 6000 token 长提示词插队：不切块时所有 Decode 用户卡顿 648ms；切成 256 一块后最长卡顿 34ms |

## 代码结构

- `BlockAllocator`：空闲链表 + 引用计数（≈ 操作系统的物理页框管理）
- `PagedKVCache`：物理 KV 池 `[num_blocks, 16, d]` + 每个序列一张块表；`append` 里实现按需分配与写时复制；`attention` 按块表逐块读取并用 Online Softmax 累加

这就是 vLLM 的 `BlockManager` 与 PagedAttention kernel 的核心逻辑，去掉了 GPU 与多层细节。

## 动手练习

1. 给 `BlockAllocator` 加上 LRU 淘汰：请求结束后块不立即释放而是进入"可回收缓存"，把实验 4 的前缀缓存和实验 2 的分配器真正接在一起。
2. 实现抢占：显存不足时选择最晚到达的请求，分别实现"换出到 CPU"与"丢弃后重算"，用第 8 篇的耗时模型比较两者代价（提示：PCIe 带宽约 25 GB/s，Prefill 的算力受限耗时见实验 5）。
3. 把块大小改成 1、16、128，观察实验 2 的浪费率与"块表长度"之间的权衡（块太小，块表和 kernel 里的间接寻址开销变大）。
4. （阅读）对照 vLLM 源码中 KV Cache 管理与调度器相关的模块（在官方文档 Design 部分可以找到架构说明），找到与本文件中每个函数对应的地方。
