# Lab 09：FlashAttention 与 FlashDecoding

> 对应理论：[第 9 篇：FlashAttention——Online Softmax 的严格推导与 IO 复杂度](../zero_to_hero_tutorial/09_FlashAttention：OnlineSoftmax的严格推导与IO复杂度.md)
> 前置：第 8 篇（存储层级、算子融合）。
> 依赖：NumPy；实验 5 可选 PyTorch + GPU。

## 运行

```bash
.venv/bin/python lab09_flash_attention/flash_attention_numpy.py
```

| 实验 | 验证的结论 |
| :--- | :--- |
| 1 | Online Softmax 一遍扫描 = 三遍扫描；打印每一步的修正因子 $e^{m_{old}-m_{new}}$ |
| 2 | 任意块大小的 FlashAttention（含因果掩码、整块跳过）与朴素注意力误差 ~1e-15 |
| 3 | 访存量模型：节省倍数 ≈ $M/d^2$，是**与 N 无关的常数**；随 N 增长的收益是 $O(N^2) \to O(N)$ 的显存 |
| 4 | FlashDecoding：KV 切 1/4/16/64 段分别计算再合并，结果一致；合并满足结合律 |
| 5 | GPU 实测：朴素注意力 vs `F.scaled_dot_product_attention` |

## 本机（RTX 5060 Ti）实验 5 结果

> ⏱️ 以下计时类数字是本机某一次运行的**量级示例**，前提是显卡空闲、已预热；GPU 降频或有其它负载时可能慢 40% 以上。请看**趋势和比例**，以你自己运行的输出为准。

| N | 朴素 | 朴素额外显存 | SDPA | SDPA 额外显存 |
| :--- | :--- | :--- | :--- | :--- |
| 1024 | 0.67 ms | 88 MB | 0.12 ms | 2 MB |
| 4096 | 14.4 ms | 1.3 GB | 0.91 ms | 8 MB |
| 8192 | 57.3 ms | 5.1 GB | 3.3 ms | 16 MB |
| 16384 | **2309 ms** | **20 GB** | 12.6 ms | 33 MB |

最后一行的朴素实现需要 20GB，超过了 16GB 显存却没有报 OOM，而是慢了 180 倍：Windows/WSL 的驱动把溢出部分放进了**系统内存**，通过 PCIe 访问。这是一个很好的提醒——"能跑"不等于"跑在显存里"。

## 动手练习

1. 在 `flash_attention` 里加上对 GQA 的支持：多个 Q 头共享同一组 K/V 块时，同一个 K/V 块读一次就能服务多个头（对照第 8 篇 §4.3）。
2. 实现 FlashAttention-1 的循环顺序（外层 K/V、内层 Q），需要把每个 Q 块的 $m, \ell$, acc 写回"显存"再读出来；统计访存量，理解 FlashAttention-2 为什么交换循环。
3. 把实验 4 的 `partial_attention` 改成读取**不连续**的 KV 块（给一个 block table，见 Lab 12b），这就是 PagedAttention kernel 的核心逻辑。
4. （进阶）跟着 [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) 写一个真正在 GPU 上运行的 FlashAttention 前向，与 SDPA 对比速度。
