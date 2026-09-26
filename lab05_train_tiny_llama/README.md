# Lab 05：亲手训练一个迷你 Llama，再在它身上做推理

> 对应理论：第 4 篇（注意力公式）、第 6 篇（KV Cache）、第 7 篇（采样）、第 3 篇（交叉熵 / 困惑度 / Adam）、第 5 篇（RMSNorm、RoPE、SwiGLU、GQA、Pre-LN）
> 前置：完成 Lab 03（你已经亲手写过反向传播，所以这里可以放心用 `loss.backward()`）。
> 依赖：PyTorch（见根目录 `requirements.txt`，RTX 50 系必须装 cu128 版本）。

## 为什么需要这个模块？

`lab04_transformer_microscope` 和 `lab06_kv_cache` 用的都是**随机权重**，输出没有任何意义。
这里你会得到一个**真正学过东西**的模型（虽然很小），然后在它身上验证推理理论——这样"KV Cache 输出完全一致""温度越高越胡说"这些结论就不再是抽象描述。

## 文件

| 文件 | 内容 |
| :--- | :--- |
| [`model.py`](model.py) | 约 150 行的 Llama 架构：RMSNorm、RoPE（半分配对，与 HF 一致）、GQA、因果掩码、SwiGLU、Pre-LN、KV Cache。每个组件的注释都标注了对应的教程章节 |
| [`train.py`](train.py) | 用本仓库的教程 Markdown 做语料，字符级训练；包含"初始损失 ≈ ln V"检查、验证集 PPL、early stopping |
| [`generate.py`](generate.py) | 朴素生成 vs KV Cache 逐 token 对拍；用真实缓存张量验证显存公式；逐步计时；采样策略对比 |

## 运行

```bash
.venv/bin/python lab05_train_tiny_llama/train.py      # GPU 约 30 秒
.venv/bin/python lab05_train_tiny_llama/generate.py
```

## 你会看到什么

以下结论都由脚本自己**断言**（不满足会直接报错），具体数值以运行输出为准：

- 初始损失 ≈ ln(V) ——初始化正确。脚本会打印实测值和理论值，并断言两者相差 < 0.5。注意 V（字表大小）会随仓库文档的增删而变化，因为语料就是本仓库的文档
- 验证集 PPL：1500 → 约 16~20（语料就是本仓库的文档，文档越多效果越好），随后**过拟合**：训练损失一路下降（默认 1000 步结束时约 0.5，训练更久会继续逼近 0），验证损失却回升。3.5M 参数 vs 十几万字符，数据量只有 Chinchilla 比例的千分之一，模型开始背书——这正是第 5 篇讲规模定律时说的"参数和数据要匹配"
- 朴素生成与 KV Cache 生成 400 个 token **完全一致**，logits 最大差 ~1e-5
- KV Cache 真实字节数 = 公式 $2 \times L \times n_{kv} \times d_{head} \times b \times T$，**逐字节相等**
- 每步耗时（本机一次运行）：

> ⏱️ 以下计时类数字是本机某一次运行的**量级示例**，前提是显卡空闲、已预热；GPU 降频或有其它负载时可能慢 40% 以上。请看**趋势和比例**，以你自己运行的输出为准。

| 上下文长度 | 朴素 (GPU) | Cache (GPU) | 朴素 (CPU) | Cache (CPU) |
| :--- | :--- | :--- | :--- | :--- |
| 11 | 3.1 ms | 3.3 ms | 3.4 ms | 1.9 ms |
| 210 | 4.1 ms | 3.1 ms | 10.6 ms | 3.4 ms |
| 409 | 3.9 ms | 4.7 ms | 37.2 ms | 2.5 ms |

CPU 上完美呈现"朴素线性变慢、缓存持平"；GPU 上两者差不多——因为模型太小，**耗时被 kernel 启动等固定开销主导**（第 8 篇的 overhead 区间）。这是真实世界的现象，不是实验失败：它解释了为什么推理引擎要用 CUDA Graph，以及为什么 Lab 07b 要换成真实的大模型再测。

## 动手练习

1. 把 `n_kv_head` 改成 8（即 MHA）重新训练，比较参数量、KV Cache 大小和验证 PPL。
2. 在 `model.py` 里把 RoPE 去掉（`apply_rope` 直接返回 `x`）重新训练，看验证 PPL 变差多少——对应 `lab04_transformer_microscope/02` 里"没有位置编码就是词袋"的实验。
3. 把 `SwiGLU` 换成 `ReLU(x W1) W2`，保持参数量相同（隐藏维度要改成多少？见第 5 篇 §5.3）。
4. 生成长度超过训练时的 `block=128` 后，文本质量有没有明显下降？这和第 5 篇 RoPE 外推的讨论有什么关系？
5. （进阶）把 `Attention.forward` 里的 `torch.cat` 缓存改成**预分配**一块 `[B, n_kv, max_len, hd]` 的张量、按位置写入，比较速度。这就是从"动态拼接"走向 PagedAttention（Lab 12b）的第一步。

## 下一步

[Lab 07b：不用 transformers 的模型类，亲手实现 Qwen2.5 的前向传播并与官方逐位对拍](../lab07b_qwen_from_scratch/)
