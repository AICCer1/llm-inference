# 第 5 篇：从原版 Transformer 到 Llama / DeepSeek：每一处改动的历史动机

> **阅读前须知**：第 1 篇讲到"2017 年 Transformer 横空出世"就结束了，第 4 篇讲的是 Transformer 的公式，Lab 04 里的代码却已经是 **RMSNorm + RoPE + SwiGLU**——这些东西 2017 年的原版 Transformer 里一个都没有。它们是怎么来的？本篇补上两段缺失的历史：
> 1. **Transformer 之前**：注意力机制到底是为了解决什么问题被发明的？（它并不是 2017 年凭空出现的）
> 2. **Transformer 之后**：从原版到 Llama-3、Qwen2.5、DeepSeek-V3，每一处改动**被什么问题逼出来**，以及它**对推理意味着什么**。
>
> **前置**：第 1、3、4 篇。特别是第 3 篇的"加法通路"和"方差控制"，本篇会反复用到。

---

## 🧭 一、注意力的真正起源（2003 → 2017）

### 1. 2003：第一个神经语言模型（Bengio NPLM）
第 1 篇说 N-gram 的死穴是"苹果"和"梨"毫无关系。2003 年 Bengio 等人提出**神经概率语言模型**：
$$ P(w_t \mid w_{t-n+1}, \dots, w_{t-1}) = \text{softmax}\big(b + W x + U \tanh(d + H x)\big), \qquad x = [e_{t-n+1}; \dots; e_{t-1}] $$
（$x$ 是前 $n-1$ 个词向量的拼接；$U\tanh(d + Hx)$ 是一个单隐藏层 MLP；$Wx$ 是论文特意加入的**从输入直连输出**的线性项，可以设为 0。）
- 每个词先查表得到一个**可学习的向量** $e$（**这就是词嵌入的真正起点**，比 Word2Vec 早 10 年）；
- 相似的词学到相似的向量，于是"我吃了苹果"的统计经验能**泛化**到"我吃了梨"。
- 但它仍然只看**固定窗口**的 $n-1$ 个词，并且 softmax 覆盖整个词表，在当时的硬件上训练极慢。

之后 2010 年 Mikolov 用 RNN 做语言模型（RNNLM）去掉了固定窗口；2013 年 Word2Vec 把"只学词向量"这件事做到极致地便宜。

### 2. 2014：Seq2Seq 与"信息瓶颈"
Sutskever 等人用两个 LSTM 做翻译：**编码器**读完整个源句子，把它压成**一个固定长度的向量** $c$；**解码器**从 $c$ 出发逐词生成译文。
问题马上暴露：无论源句子 5 个词还是 50 个词，都要塞进同一个几百维的向量里。Cho 等人（2014）的实验显示**句子越长，翻译质量下降越明显**。这个固定长度向量就是**信息瓶颈**。

### 3. 2014：Bahdanau 注意力——"别压缩了，每一步回头看"
Bahdanau、Cho、Bengio 的想法是：解码器生成第 $t$ 个词时，**不要只看那个压缩向量，而是回头看编码器每个位置的隐藏状态 $h_j$，按需加权**：
$$ e_{tj} = a(s_{t-1}, h_j), \qquad \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_k \exp(e_{tk})}, \qquad c_t = \sum_j \alpha_{tj} h_j $$

**请对照第 4 篇的注意力公式**：
| Bahdanau 2014 | Transformer 2017 |
| :--- | :--- |
| 解码器当前状态 $s_{t-1}$（"我现在要翻译什么"） | Query |
| 编码器各位置 $h_j$（用来算匹配分数） | Key |
| 编码器各位置 $h_j$（被加权求和的内容） | Value |
| 打分函数 $a(\cdot)$ 是一个小 MLP（加性注意力） | 打分函数是点积 $q \cdot k / \sqrt{d_k}$ |

2015 年 Luong 等人把打分函数换成**点积**（乘性注意力）——更便宜，而且整批计算恰好是一次矩阵乘，GPU 最擅长。**Q/K/V 这三个角色、softmax 加权求和，在 2014 年就已经全部出现了。**

### 4. 2017：Transformer 的真正创新是"只要注意力"
既然注意力这么有用，而 RNN 又串行、又会遗忘（第 1、3 篇），Vaswani 等人问：**能不能把 RNN 整个扔掉，只用注意力？**
- **自注意力**：Query、Key、Value 都来自同一个序列自身（而不是解码器看编码器）；
- **位置编码**：扔掉 RNN 后，模型丧失了顺序感（`lab04_transformer_microscope/02_rope_and_attention_microscope.py` 实验 1 证明了这一点），必须显式注入位置；
- **多头**：多组独立的 $W_q, W_k, W_v$，让不同的头关注不同类型的关系；
- **缩放点积**：除以 $\sqrt{d_k}$（第 4 篇的推导）。

原版 Transformer 是**编码器-解码器**结构，用来做翻译，6 层编码器 + 6 层解码器，$d_{\text{model}} = 512$，FFN 用 ReLU，**Post-LN**，**正弦位置编码**。

---

## ✂️ 二、分词（Tokenization）：为什么"词表大小"会影响推理速度

### 1. 为什么不按"词"或"字"切？
- **按词切**：词表爆炸（英文有各种变形，中文没有天然空格），总会遇到没见过的词（OOV, Out-Of-Vocabulary）只能标记为 `<unk>`；
- **按字符 / 字节切**：永远不会 OOV，但序列变得很长——注意力是 $O(n^2)$，而且**Decode 步数 = token 数**，每多一个 token 就要多读一遍全部权重（第 6 篇）。

### 2. BPE：从数据压缩借来的算法
2016 年 Sennrich 等人把 1994 年的数据压缩算法 **BPE（Byte Pair Encoding）** 用到翻译上：
1. 从单个字符（或字节）出发作为初始词表；
2. 统计语料里**最常相邻出现的一对符号**，合并成一个新符号加入词表；
3. 重复第 2 步，直到词表达到目标大小（例如 32000、128000、151936）。

结果：常见词是一个 token，罕见词被拆成几个子词，**永远不会 OOV**。GPT-2 进一步在**字节**上做 BPE（Byte-level BPE），任何 UTF-8 文本都能编码。

### 3. 词表大小是一个推理工程上的权衡
| 词表更大 | 词表更小 |
| :--- | :--- |
| ✅ 同样的文本 token 更少 → Decode 步数更少、KV Cache 更小 | ✅ Embedding 与 LM Head 更小 |
| ❌ Embedding 表和 LM Head 更大：LM Head 每个 token 都要算 $d \times V$ 的矩阵乘 | ❌ 同样文本 token 更多 |

例子：Qwen2.5-0.5B 的 $V = 151936$、$d = 896$，**光是词嵌入矩阵就有 1.36 亿参数，占全模型约 28%**（Lab 07b 会打印出来）。Llama-2 词表只有 32000，对中文很不友好（一个汉字常被切成多个 token）；Llama-3 扩到 128256、Qwen 用 151936，中文同样的内容 token 数显著减少——**这直接等于中文推理变快、变便宜**。

> 🔗 深入：Karpathy 的视频 *Let's build the GPT Tokenizer* 与代码 [minbpe](https://github.com/karpathy/minbpe)。

---

## 🏛️ 三、三条路线：为什么最后是"只有解码器"的 GPT 赢了？

| 路线 | 代表 | 训练目标 | 能力特点 |
| :--- | :--- | :--- | :--- |
| 仅编码器 | BERT（2018） | 完形填空（遮住 15% 的词，双向看上下文去猜） | 理解任务极强，但**不能自回归生成** |
| 编码器-解码器 | 原版 Transformer、T5（2019） | 输入 → 输出 | 适合翻译、摘要等"有明确输入输出"的任务 |
| 仅解码器 | GPT 系列、Llama、Qwen、DeepSeek | **预测下一个词** | 什么文本都能当训练数据；生成与理解统一成"续写" |

主流的解释（不是定理，而是历史经验）：
1. **目标最简单、数据最多**：任何一段文本都是"预测下一个词"的训练数据，无需标注；
2. **一切任务都能写成续写**：GPT-3（2020）展示了**上下文学习（In-Context Learning）**——在提示里给几个例子，不改参数就能做新任务；
3. **规模化最顺**：同样算力下，仅解码器结构简单、易于扩展；
4. **对推理友好**：只有一种因果注意力，KV Cache 结构统一、实现简单。

---

## 📈 四、规模定律：为什么"推理成本"成了整个行业的主问题

- **2019 GPT-2（1.5B）**：发现模型够大后，不微调也能做一些任务（zero-shot）。
- **2020 Kaplan 等 *Scaling Laws***：损失随参数量 $N$、数据量 $D$、算力 $C$ 呈**幂律**下降，例如 $L(N) \propto N^{-0.076}$。只要按规律加大规模，效果就可预测地变好。
- **2020 GPT-3（175B）**：上下文学习能力涌现。
- **2022 Chinchilla（Hoffmann 等）**：在**固定训练算力**下，最优做法是参数量与训练 token 数同比例增长，约 **每个参数 20 个 token**。此前的大模型普遍"参数太多、数据太少"。
- **2022 InstructGPT → ChatGPT**：用指令微调 + RLHF 让模型"听话"，大模型从研究品变成亿级用户产品。

**关键转折**：Chinchilla 只优化了**训练**成本。但一个模型训练一次，却要被调用成千上万亿次。2023 年的 Llama 明确把目标改成"**给定推理预算下效果最好**"——宁可用远超 Chinchilla 比例的数据去**过度训练一个小模型**（Llama-1-7B 用了 1T token，Llama-3-8B 用了 15T token，约为 Chinchilla 比例的近百倍）。
**从这一刻起，推理效率成为模型设计本身的一等公民**：下面第五节的 MQA/GQA/MLA、MoE、词表扩大，全部带着"推理更便宜"的动机。

---

## 🔧 五、从原版 Transformer 到现代 LLM：逐个改动的"为什么"

### 1. Post-LN → Pre-LN：把归一化挪出"高速公路"
- **原版 Post-LN**：$x_{l} = \text{LN}(x_{l-1} + F(x_{l-1}))$。LN 挡在残差路径上，第 3 篇说的"$I$ 直通路径"被每一层的 LN 打断，梯度必须穿过 $L$ 个 LN 的雅可比。深层时训练不稳定，必须用学习率预热（warmup）小心伺候。
- **Pre-LN**：$x_{l} = x_{l-1} + F(\text{LN}(x_{l-1}))$。残差路径上**什么都没有**，梯度可以原封不动流回第一层。Xiong 等（2020）从理论上分析了这一点；实际上 GPT-2（2019）就已经改成了 Pre-LN。
- **代价**：Pre-LN 的残差流数值会随层数累积变大，所以最后要再加一个 **Final Norm**（你在 `lab04_transformer_microscope` 里看到的 Step 8）。

### 2. LayerNorm → RMSNorm：去掉"减均值"
$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta \qquad \longrightarrow \qquad \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma $$
Zhang & Sennrich（2019）发现 LayerNorm 起作用的主要是"**缩放**"（re-scaling），"**减均值**"（re-centering）贡献很小。RMSNorm 少一次求均值的归约、没有 $\beta$，效果相当。
**对推理的意义**：归一化是典型的**访存受限**小算子（读一遍、写一遍，算得很少），少一次归约就少一次同步，也更容易和前后算子**融合（fuse）**成一个 kernel（第 8 篇）。

### 3. ReLU → GELU → SwiGLU：门控前馈网络
- 原版 FFN：$\text{ReLU}(x W_1) W_2$，隐藏维度 $4d$。
- BERT / GPT-2 换成更平滑的 **GELU**。
- Shazeer（2020）系统比较了多种**门控线性单元（GLU）**变体，SwiGLU 在同等计算量下效果最好：
$$ \text{SwiGLU}(x) = \big(\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big) W_{\text{down}} $$
  门控的直觉：一路计算"内容"，另一路计算"让多少内容通过"，两者**相乘**，比单纯的逐元素非线性表达力更强。有趣的是，论文作者自己写道：*"我们不解释这些架构为何有效，把它们的成功和其他一切一样，归功于上天的仁慈。"*——深度学习很多改动的"为什么"首先是**实验结果**，理论解释往往滞后。
- **一个你在配置文件里会看到的数字**：SwiGLU 有 3 个矩阵而不是 2 个。为了保持参数量不变，隐藏维度从 $4d$ 缩成 $\frac{2}{3} \times 4d = \frac{8}{3}d$。Llama-7B 的 $d = 4096$，$\frac{8}{3} \times 4096 \approx 10923$，向上取整到 256 的倍数就是 **11008**——这就是 Llama 配置里那个"奇怪数字"的来历。

### 4. 绝对位置编码 → RoPE：为了"外推"与"相对位置"
- 原版用正弦函数的绝对位置编码，加在输入上；GPT-2 改成**可学习的**绝对位置向量——但训练时最长 1024，就**根本没有**第 1025 个位置的向量，无法外推。
- T5（相对位置偏置）、ALiBi（按距离线性惩罚注意力分数）走向了"相对位置"。
- **RoPE**（苏剑林等，2021）：对 Q、K 按位置旋转，内积只依赖相对距离（第 4 篇已证明），**没有额外参数**。

**RoPE 对推理的一个关键影响**：K 在**写入 KV Cache 之前**就已经按它自己的位置旋转好了。之后无论新 token 走到哪个位置，缓存里的 K 都不用再动——只需要旋转新来的 q 和 k。Lab 07b 里你会亲手实现这一点。

**长上下文扩展**：RoPE 的频率 $\theta_i = b^{-2i/d}$ 中的底数 $b$ 决定了"最慢的那一对分量转多快"。训练长度为 $L$ 时，低频维度只转过了很小的角度；推理时位置一旦超过 $L$，这些维度会转到**训练中从未见过的角度**（分布外），模型行为随之失控——这才是需要扩展方法的原因（RoPE 对"匹配信号"的期望衰减是另一回事，见第 4 篇 §6）。要把训练时的 4K 上下文扩到 32K、128K，常见做法是**调整频率**：位置插值（PI, 2023）、NTK-aware 缩放、YaRN（2023）。这也是为什么 Llama-3 把 $b$ 从 10000 调到 500000，Qwen2.5 用 1000000。

### 5. MHA → MQA → GQA → MLA：一部"为 KV Cache 减肥"的历史
这一条演进**完全由推理驱动**，是本篇和第 6 篇之间最直接的桥梁。

| 方案 | 年份 | KV 头数 | 每 token 每层 KV 存储 | 动机与代价 |
| :--- | :--- | :--- | :--- | :--- |
| MHA 多头注意力 | 2017 | $= n_h$ | $2 n_h d_h$ | 原版 |
| **MQA** 多查询注意力 | 2019 | $1$ | $2 d_h$ | Shazeer 的论文标题就叫 *One Write-Head is All You Need*，**明确为了解决增量解码的显存带宽瓶颈**。代价：质量下降、训练不稳 |
| **GQA** 分组查询注意力 | 2023 | $g$（如 8） | $2 g d_h$ | Ainslie 等人的折中：每 $n_h / g$ 个 Q 头共享一组 KV。质量接近 MHA，KV 缩小 $n_h / g$ 倍。Llama-2-70B、Llama-3、Qwen2.5 均采用 |
| **MLA** 多头潜在注意力 | 2024 | — | $d_c + d_h^R$（如 512 + 64） | DeepSeek-V2：不存 K、V 本身，而是存一个低秩的**压缩潜向量** $c^{KV}$，用时再"解压"；解压矩阵还能代数上吸收进 $W_q$ 与 $W_o$。RoPE 与低秩压缩不兼容，所以额外存一个 64 维的带位置小 key |

用数字感受一下（BF16，每 token，全部层）：
- Llama-3-8B（32 层，GQA-8，$d_h = 128$）：$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ 字节 = **128 KB**
- DeepSeek-V3（671B 总参数，61 层，MLA）：$(512 + 64) \times 61 \times 2 \approx 70{,}272$ 字节 ≈ **69 KB**
- 如果 DeepSeek-V3 用 MHA（128 头 × 128 维）：$2 \times 128 \times 128 \times 61 \times 2 \approx$ **3.8 MB**

一个 671B 的模型，每 token 的 KV Cache 比 8B 的 Llama-3 还小。**这就是"模型架构为推理服务"的最佳例证。**

### 6. 稠密 → MoE（混合专家）：参数多 ≠ 算得多
把每层的 FFN 换成 $E$ 个"专家" FFN，再加一个**路由器**：每个 token 只挑得分最高的 $k$ 个专家计算（例如 $E = 256$ 选 $k = 8$）。
- Mixtral 8x7B：总参数约 47B，每 token 激活约 13B；
- DeepSeek-V3：总参数 671B，每 token 激活 37B。

**对推理的意义（很反直觉）**：
- **算力**只跟"激活参数"有关 → Prefill 很便宜；
- **显存**要装下**全部**参数 → 必须多卡；
- **Decode 带宽**：batch 很小时每步只读被选中专家的权重；但 batch 一大，几乎所有专家都会被某个 token 选中，每一步又要读全部权重——于是出现了**专家并行（EP）**等专门的部署方案（第 13 篇）。

### 7. 其它细节改动
- **去掉偏置项**：Llama 系列去掉了线性层的 bias（Qwen2 保留了 Q/K/V 投影的 bias，Lab 07b 实现时要注意）。
- **词嵌入与 LM Head 共享权重（tied embeddings）**：小模型里词表矩阵占比极大，共享可省下几亿参数。Qwen2.5-0.5B 就是共享的。
- **滑动窗口注意力**（Mistral-7B, 2023）：每个 token 只看最近 $W$ 个 token，KV Cache 大小封顶。

---

## 🗺️ 六、一张总表：同一个"Transformer"，七年里变了什么

| 组件 | 原版 Transformer（2017） | GPT-2/3（2019-20） | Llama-1（2023） | Llama-3 / Qwen2.5（2024） | DeepSeek-V3（2024） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 结构 | 编码器-解码器 | 仅解码器 | 仅解码器 | 仅解码器 | 仅解码器 |
| 归一化位置 | Post-LN | **Pre-LN** | Pre-LN | Pre-LN | Pre-LN |
| 归一化类型 | LayerNorm | LayerNorm | **RMSNorm** | RMSNorm | RMSNorm |
| FFN 激活 | ReLU | GELU | **SwiGLU** | SwiGLU | SwiGLU（**MoE**） |
| 位置编码 | 正弦绝对 | 可学习绝对 | **RoPE** | RoPE（大底数） | RoPE（解耦 + YaRN） |
| 注意力 | MHA | MHA | MHA | **GQA** | **MLA** |
| 词表 | ~37K（BPE） | 50257（字节 BPE） | 32000 | 128256 / 151936 | 129280 |

> 🎯 **读完本篇你应该能做到**：打开任意一个模型的 `config.json`（例如 Hugging Face 上 Qwen2.5 的配置），逐个字段说出它是什么、为什么有它、对推理的显存和速度有什么影响。Lab 07b 的第一个任务就是这个。

---

## ✅ 本篇自测

1. Bahdanau 注意力要解决 Seq2Seq 的什么问题？它和 Transformer 的注意力，Q/K/V 分别对应什么？
2. 为什么 Transformer 必须有位置编码，而 RNN 不需要？
3. 为什么 Llama-7B 的 FFN 中间维度是 11008？
4. Pre-LN 比 Post-LN 好训练，用第 3 篇的"加法通路"解释。
5. GQA-8 的 70B 模型比 MHA 版本省多少 KV Cache？为什么说 MQA 的动机来自推理而不是训练？
6. MoE 模型"参数 671B、激活 37B"，在 batch=1 与 batch=256 时，Decode 每一步需要读的权重量分别大约是多少？
7. Llama-3 为什么要用远超 Chinchilla 最优比例的数据训练 8B 模型？

---

## 📚 外部资源

**入门可视化**
- Jay Alammar：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)、[Visualizing A Neural Machine Translation Model (Seq2Seq + Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- Lilian Weng：[Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)（从 Bahdanau 到 Transformer 的注意力家谱）
- 3Blue1Brown：[神经网络系列](https://www.3blue1brown.com/topics/neural-networks) 第 5~7 集（GPT、注意力、MLP 如何存储知识）

**跟着写代码**
- Karpathy：*Let's build GPT: from scratch, in code, spelled out* 与 *Let's build the GPT Tokenizer*（[Zero to Hero](https://karpathy.ai/zero-to-hero.html) 第 7、8 集）
- Harvard NLP：[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)（原版论文逐段对照代码）
- Sebastian Raschka：[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（含 GPT → Llama → Qwen 的逐步改造笔记本）

**中文深度**
- 苏剑林（RoPE 作者）博客：[Transformer 升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265)

**论文清单（按本篇出现顺序）**
- Bengio et al. 2003 [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Mikolov et al. 2013 [Word2Vec](https://arxiv.org/abs/1301.3781)
- Sutskever et al. 2014 [Seq2Seq](https://arxiv.org/abs/1409.3215)；Cho et al. 2014 [On the Properties of NMT](https://arxiv.org/abs/1409.1259)
- Bahdanau et al. 2014 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)；Luong et al. 2015 [Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025)
- Vaswani et al. 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Sennrich et al. 2016 [BPE for NMT](https://arxiv.org/abs/1508.07909)
- Devlin et al. 2018 [BERT](https://arxiv.org/abs/1810.04805)；Raffel et al. 2019 [T5](https://arxiv.org/abs/1910.10683)
- Radford et al. 2018 [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)；2019 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)；Brown et al. 2020 [GPT-3](https://arxiv.org/abs/2005.14165)
- Kaplan et al. 2020 [Scaling Laws](https://arxiv.org/abs/2001.08361)；Hoffmann et al. 2022 [Chinchilla](https://arxiv.org/abs/2203.15556)；Ouyang et al. 2022 [InstructGPT](https://arxiv.org/abs/2203.02155)
- Xiong et al. 2020 [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)；Zhang & Sennrich 2019 [RMSNorm](https://arxiv.org/abs/1910.07467)
- Hendrycks & Gimpel 2016 [GELU](https://arxiv.org/abs/1606.08415)；Shazeer 2020 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Su et al. 2021 [RoFormer](https://arxiv.org/abs/2104.09864)；Press et al. 2021 [ALiBi](https://arxiv.org/abs/2108.12409)；Chen et al. 2023 [Position Interpolation](https://arxiv.org/abs/2306.15595)；Peng et al. 2023 [YaRN](https://arxiv.org/abs/2309.00071)
- Shazeer 2019 [MQA](https://arxiv.org/abs/1911.02150)；Ainslie et al. 2023 [GQA](https://arxiv.org/abs/2305.13245)；DeepSeek-AI 2024 [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- Fedus et al. 2021 [Switch Transformer](https://arxiv.org/abs/2101.03961)；Jiang et al. 2024 [Mixtral](https://arxiv.org/abs/2401.04088)
- Touvron et al. 2023 [Llama](https://arxiv.org/abs/2302.13971)、[Llama 2](https://arxiv.org/abs/2307.09288)；Llama Team 2024 [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

---

**上一篇**：[第 4 篇：Transformer 核心公式彻底推导：每一个符号从何而来](04_Transformer核心公式的彻底推导：每一个符号从何而来.md) ｜ **下一篇**：[第 6 篇：什么是推理？自回归生成全生命周期与 KV Cache 代数证明](06_什么是推理：自回归生成全生命周期与KVCache代数证明.md) ｜ **总路线**：[LEARNING_PATH.md](../LEARNING_PATH.md)
