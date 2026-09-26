# Lab 07a：采样策略显微镜

> 对应理论：[第 7 篇：Logits、温度系数极限证明与采样算法](../zero_to_hero_tutorial/07_大模型的决策大脑：Logits、温度系数极限证明与采样算法.md)
> 依赖：纯 Python 标准库。

## 运行

```bash
python3 lab07a_sampling/sampling_strategies.py
```

用一组固定的 Logits，逐一对比：原始分布、低温（T=0.2）、高温（T=2.0）、Top-K、Top-P、重复惩罚，打印每种策略处理后的概率分布条形图。

## 速查

模型 LM Head 输出的不是概率，而是一堆未经归一化的实数值，称为 **Logits**（例如 `[2.1, -0.5, 8.4, 0.1]`）。
- **贪婪搜索（Greedy Search）**：
  $$\text{next\_token} = \arg\max(\text{logits})$$
  直接选得分最高的那一个（确定性强，但容易重复或死板）。
- **温度系数（Temperature $T$）**：
  $$\text{probs} = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$
  - $T \to 0$：分布极度尖锐，退化为贪婪搜索；
  - $T > 1$：分布平缓，低分词也有机会被选中，回答更具创意但更容易胡说八道；
- **Top-P（核采样 Nucleus Sampling）**：
  按概率从高到低排序累加，只保留累计概率达到 $P$（如 $0.9$）的头部候选词，截断长尾无意义低概率词，在多样性与合理性之间取得最佳平衡。

## 下一步

- 在**真正训练过的模型**上看采样效果：[Lab 05](../lab05_train_tiny_llama/) 的 `generate.py` 实验 4；
- 手写重复惩罚并与 Hugging Face 官方实现逐 token 对拍：[Lab 07b](../lab07b_qwen_from_scratch/) 实验 5。
