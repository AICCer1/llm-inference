# Lab 03：从零手写自动求导（Autograd）

> 对应理论：[第 3 篇：神经网络如何学习：导数、链式法则、反向传播与交叉熵](../zero_to_hero_tutorial/03_神经网络如何学习：导数、链式法则、反向传播与交叉熵.md)
> 前置：读完第 3 篇第一 ~ 六节。
> 依赖：纯 Python 标准库。

## 为什么要自己写一遍？

PyTorch 的 `loss.backward()` 是整个深度学习最大的"黑盒"。只要你亲手写过一次，就会明白：
- 反向传播 = **链式法则 + 拓扑排序 + 梯度累加**，没有任何魔法；
- 为什么每一步训练前必须 `zero_grad()`（梯度是 `+=` 累加的）；
- 为什么训练要保存激活值（`_backward` 闭包里用到了前向时的 `self.data`）；
- 为什么推理时可以用 `torch.no_grad()` / `torch.inference_mode()` 省掉大量显存（不建图、不存激活）。

## 运行

```bash
python3 lab03_autograd_from_scratch/micrograd_from_scratch.py
```

| 实验 | 验证的结论 | 对应章节 |
| :--- | :--- | :--- |
| 1 | 手算表格 dL/dw = −12, dL/db = −6 | 第 3 篇 §2.3 |
| 2 | 自动求导 = 数值差分（梯度检验） | 第 3 篇 §3 |
| 3 | Softmax 雅可比 $p_i(\delta_{ij}-p_j)$；交叉熵梯度 $p - y$；logits 放大后梯度坍缩 | 第 3 篇 §6、第 4 篇 §3 |
| 4 | $\partial L/\partial W = X^T G$，$\partial L/\partial X = G W^T$ | 第 3 篇 §4 |
| 5 | Sigmoid 连乘梯度消失 vs 残差畅通 | 第 3 篇 §7、第 4 篇 §5 |
| 6 | 单层学不会 XOR，两层可以 | 第 1 篇 §4.2、第 0 篇族谱 |

## 动手练习（做完才算过关）

1. 给 `Value` 加一个 `silu()` 算子（$x\sigma(x)$），自己推导导数，并用实验 2 的方法做梯度检验。
2. 把实验 6 的激活函数从 `tanh` 换成 `sigmoid`，学习率不变，观察收敛速度变化，并用第 3 篇 §7 的导数上界解释。
3. 把实验 5 的深度改成 30，看纯 Sigmoid 的梯度变成多少。
4. （进阶）跟着 [Karpathy 的 micrograd 视频](https://karpathy.ai/zero-to-hero.html) 实现 `Neuron / Layer / MLP` 类，并训练一个二分类的"月牙"数据集。

## 学完之后

你已经可以放心使用 `torch.autograd` 了——它做的事和这 150 行完全一样，只是算子换成了张量、实现换成了 C++/CUDA。
下一步：[第 5 篇](../zero_to_hero_tutorial/05_从原版Transformer到Llama与DeepSeek：每一处改动的历史动机.md) → [Lab 05：亲手训练一个迷你 GPT](../lab05_train_tiny_llama/)。
