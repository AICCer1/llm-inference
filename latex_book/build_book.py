#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型推理系统与核心原理：从零基础到工业实战
全自动 Markdown -> LaTeX 专著转换与排版引擎 (v5 终极完美版)
"""

import os
import re
import sys
import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))
CHAPTERS_DIR = os.path.join(OUTPUT_DIR, "chapters")
LABS_DIR = os.path.join(OUTPUT_DIR, "labs")
CODE_DIR = os.path.join(OUTPUT_DIR, "code")

os.makedirs(CHAPTERS_DIR, exist_ok=True)
os.makedirs(LABS_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 预置高清精细排版 TikZ 架构图 (严格调整间距，杜绝任何重叠)
# ------------------------------------------------------------------------------
TIKZ_AI_TREE = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  base/.style={draw=primary!80, fill=lightbg, rounded corners=2mm, align=center, font=\footnotesize\sffamily, inner sep=4pt, line width=0.8pt},
  vnode/.style={base, draw=secondary!80, fill=secondary!8, minimum width=3.8cm},
  nnode/.style={base, draw=accent!80, fill=accent!8, minimum width=3.8cm},
  tform/.style={base, draw=primary, fill=primary!15, line width=1.2pt, font=\bfseries\footnotesize\sffamily, minimum width=8cm},
  arr/.style={-{Stealth[scale=0.8]}, draw=gray!70, line width=0.8pt},
  darr/.style={-{Stealth[scale=0.8]}, dashed, draw=primary, line width=0.9pt}
]

% Level 0 & 1
\node[base, minimum width=6.5cm] (p) {1957 感知机 (Perceptron)\\单神经元，无法解决 XOR 异或};
\node[base, below=0.5cm of p, minimum width=6.5cm] (mlp) {1986 多层感知机 (MLP / BP)\\反向传播与现代神经网络基石};

% Level 2: CNN vs RNN
\node[vnode, below left=0.7cm and 0.4cm of mlp] (cnn) {1998 LeNet / CNN\\局部滑窗提取空间特征};
\node[nnode, below right=0.7cm and 0.4cm of mlp] (rnn) {1980s RNN (循环网络)\\时序备忘录处理文本};

% Level 3: AlexNet vs LSTM
\node[vnode, below=0.5cm of cnn] (alex) {2012 AlexNet (GPU 革命)\\算力与深度爆发};
\node[nnode, below=0.5cm of rnn] (lstm) {1997 LSTM / 2014 GRU\\门控机制缓解长程遗忘};

% Level 4: ResNet vs Seq2Seq
\node[vnode, below=0.5cm of alex] (resnet) {2015 ResNet (何恺明)\\残差连接 $x+F(x)$ 破百层};
\node[nnode, below=0.5cm of lstm] (seq2seq) {2014 Seq2Seq + Attention\\机器翻译学会“看重点”};

% Level 5: Transformer (Centered below both branches)
\node[tform, below=1.0cm of $(resnet.south)!0.5!(seq2seq.south)$] (trans) {2017 Transformer (Google 论文)\\彻底废黜循环时序！全矩阵并行点积！统一语言与多模态};

% Level 6: Successors
\node[base, below left=0.8cm and -0.8cm of trans, minimum width=3.2cm] (bert) {2018 BERT\\双向理解完形填空};
\node[base, below=0.8cm of trans, minimum width=3.6cm, fill=primary!10, draw=primary] (gpt) {2018--2023 GPT-1$\sim$4\\单向自回归接龙};
\node[base, below right=0.8cm and -0.8cm of trans, minimum width=3.2cm] (vit) {2020 ViT\\视觉 Transformer};

% Level 7: Modern LLM Era
\node[base, below=0.6cm of gpt, fill=accent!10, draw=accent, line width=1pt, font=\bfseries\footnotesize\sffamily, minimum width=8.5cm] (llm) {2023 至今：现代大模型时代 (Llama-3, Qwen-2.5, DeepSeek)\\核心转向：推理吞吐、显存架构优化、KV Cache、量化};

% Arrows
\draw[arr] (p) -- (mlp);
\draw[arr] (mlp) -| (cnn);
\draw[arr] (mlp) -| (rnn);
\draw[arr] (cnn) -- (alex);
\draw[arr] (alex) -- (resnet);
\draw[arr] (rnn) -- (lstm);
\draw[arr] (lstm) -- (seq2seq);

\draw[darr] (resnet.south) -- ++(0,-0.4) -| node[near start, left, font=\scriptsize] {残差连接思想} ($(trans.north west)+(0.8,0)$);
\draw[darr] (seq2seq.south) -- ++(0,-0.4) -| node[near start, right, font=\scriptsize] {注意力思想飞跃} ($(trans.north east)+(-0.8,0)$);

\draw[arr] (trans) -- (bert);
\draw[arr] (trans) -- (gpt);
\draw[arr] (trans) -- (vit);
\draw[arr] (gpt) -- (llm);

\end{tikzpicture}
\caption{AI 深度学习演化科技树全景族谱}
\end{figure}
"""

TIKZ_TRANSFORMER_PIPELINE = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  stage/.style={draw=primary!80, fill=primary!5, rounded corners=1.5mm, align=center, font=\small\sffamily, inner sep=4.5pt, line width=0.8pt, minimum width=11cm},
  arr/.style={-{Stealth[scale=0.9]}, draw=primary, line width=1pt}
]
\node[stage] (s0) {原始输入文本：\texttt{"人工智能"}};
\node[stage, below=0.4cm of s0] (s1) {1. Tokenizer 分词 $\rightarrow$ Token IDs: \texttt{[102, 305]} \quad (Shape: $[2]$)};
\node[stage, below=0.4cm of s1] (s2) {2. Embedding 查表 $\rightarrow$ 稠密向量矩阵 \quad (Shape: $[2, d_{\mathrm{model}}]$)};
\node[stage, below=0.4cm of s2] (s3) {3. RMSNorm 归一化 $\rightarrow$ 消除激活值方差偏置 \quad (Shape: $[2, d_{\mathrm{model}}]$)};
\node[stage, below=0.4cm of s3] (s4) {4. 多头自注意力 (Self-Attention + RoPE 旋转位置编码) \quad (Shape: $[2, d_{\mathrm{model}}]$)};
\node[stage, below=0.4cm of s4] (s5) {5. 残差连接 + SwiGLU / MLP 非线性知识映射 \quad (Shape: $[2, d_{\mathrm{model}}]$)};
\node[stage, below=0.4cm of s5, fill=secondary!10, draw=secondary] (s6) {6. 循环堆叠 $L$ 个 Transformer 块 (重复步骤 3 $\sim$ 5)};
\node[stage, below=0.4cm of s6] (s7) {7. Final RMSNorm + LM Head 词表投影 $\rightarrow$ Logits \quad (Shape: $[2, V]$)};
\node[stage, below=0.4cm of s7, fill=accent!10, draw=accent] (s8) {8. 采样算法 (Top-P / Temperature) $\rightarrow$ 生成下一个 Token ID: 520 (\texttt{"的"})};

\draw[arr] (s0) -- (s1);
\draw[arr] (s1) -- (s2);
\draw[arr] (s2) -- (s3);
\draw[arr] (s3) -- (s4);
\draw[arr] (s4) -- (s5);
\draw[arr] (s5) -- (s6);
\draw[arr] (s6) -- (s7);
\draw[arr] (s7) -- (s8);
\end{tikzpicture}
\caption{Transformer 单步推理张量生命周期微观全貌}
\end{figure}
"""

TIKZ_SPECULATIVE_DECODING = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  actor/.style={draw=primary, fill=primary!10, rounded corners=2mm, font=\bfseries\small\sffamily, align=center, inner sep=6pt, minimum width=4cm},
  msg/.style={-{Stealth[scale=0.9]}, draw=secondary, line width=1.1pt, font=\small\sffamily},
  note/.style={draw=gray!50, fill=gray!5, rounded corners=1.5mm, font=\footnotesize\sffamily, align=center, inner sep=5pt}
]
\node[actor] (draft) {草稿小模型 ($M_{\mathrm{draft}}$)\\小而快 (Draft Model)};
\node[actor, right=5cm of draft] (target) {目标大模型 ($M_{\mathrm{target}}$)\\大而准 (Target Model)};

\coordinate[below=6.2cm of draft] (draft_end);
\coordinate[below=6.2cm of target] (target_end);

\draw[line width=1pt, gray!60, dashed] (draft) -- (draft_end);
\draw[line width=1pt, gray!60, dashed] (target) -- (target_end);

\node[note, below=0.8cm of draft] (n1) {自回归快速生成 $K$ 个候选 Token\\$[x_1, x_2, x_3, x_4]$};
\draw[msg] ($(draft.south)+(0,-1.8)$) -- node[above, font=\small\sffamily\color{secondary}] {1. 提交草稿预测序列 $[x_1, \dots, x_K]$} ($(target.south)+(0,-1.8)$);
\node[note, below=2.2cm of target] (n2) {单次并行前向传播 (Parallel Forward)\\同时计算全部 $K$ 个位置真实概率分布 $p(x)$};
\node[note, below=3.6cm of target] (n3) {逐位执行投机判定：\\接受率 $r_i = \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$};
\draw[msg] ($(target.south)+(0,-5.2)$) -- node[above, font=\small\sffamily\color{secondary}] {2. 返回接收的前缀 + 补采样调整词元} ($(draft.south)+(0,-5.2)$);

\end{tikzpicture}
\caption{投机采样（Speculative Decoding）草稿与目标模型协同序列交互}
\end{figure}
"""

TIKZ_PERCEPTRON = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=0.88, >=stealth,
  input/.style={rectangle, rounded corners=2pt, draw=secondary, fill=secondary!10, font=\sffamily\footnotesize},
  weight/.style={rectangle, rounded corners=2pt, draw=primary, fill=primary!10, font=\sffamily\footnotesize},
  neuron/.style={circle, draw=warnred, fill=warnred!10, font=\sffamily\footnotesize\bfseries, minimum size=0.9cm}
]
  \node[input] (x1) at (0, 1.5) {特征 $x_1$ (离地铁距离)};
  \node[input] (x2) at (0, 0) {特征 $x_2$ (房屋面积)};
  \node[input] (x3) at (0, -1.5) {特征 $x_3$ (是否带学区)};
  
  \node[weight] (w1) at (3.2, 1.5) {权重 $w_1$};
  \node[weight] (w2) at (3.2, 0) {权重 $w_2$};
  \node[weight] (w3) at (3.2, -1.5) {权重 $w_3$};
  
  \node[neuron] (sum) at (6.0, 0) {$\sum w_i x_i + b$};
  \node[weight, fill=tipgreen!15, draw=tipgreen] (act) at (8.5, 0) {激活函数 $f$};
  \node[input, fill=primary!15, draw=primary] (out) at (10.8, 0) {预测输出 $\hat{y}$};
  
  \draw[->, thick] (x1) -- (w1);
  \draw[->, thick] (x2) -- (w2);
  \draw[->, thick] (x3) -- (w3);
  
  \draw[->, thick] (w1) -- (sum);
  \draw[->, thick] (w2) -- (sum);
  \draw[->, thick] (w3) -- (sum);
  
  \draw[->, thick] (sum) -- (act);
  \draw[->, thick] (act) -- (out);
\end{tikzpicture}
\caption{单神经元（感知机）加权求和与非线性激活计算图}
\end{figure}
"""

TIKZ_MLP_LAYERS = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=0.9, >=stealth,
  nodeIn/.style={rectangle, rounded corners=2pt, draw=secondary, fill=secondary!10, font=\sffamily\scriptsize, inner sep=3pt},
  nodeHid/.style={circle, draw=primary, fill=primary!10, font=\sffamily\scriptsize, inner sep=2pt},
  nodeOut/.style={rectangle, rounded corners=2pt, draw=warnred, fill=warnred!10, font=\sffamily\scriptsize, inner sep=3pt}
]
  \node[font=\sffamily\footnotesize\bfseries, align=center] at (0, 2.3) {输入层\\[-0.1em]\scriptsize (Input)};
  \node[font=\sffamily\footnotesize\bfseries, align=center] at (3.5, 2.3) {隐藏层 1\\[-0.1em]\scriptsize (Hidden 1)};
  \node[font=\sffamily\footnotesize\bfseries, align=center] at (6.5, 2.3) {隐藏层 2\\[-0.1em]\scriptsize (Hidden 2)};
  \node[font=\sffamily\footnotesize\bfseries, align=center] at (10.0, 2.3) {输出层\\[-0.1em]\scriptsize (Output)};
  
  \foreach \i in {1, 2, 3} {
    \node[nodeIn] (in\i) at (0, 2.0 - \i*1.0) {输入特征 $x_\i$};
    \node[nodeHid] (h1\i) at (3.5, 2.0 - \i*1.0) {$h_{1,\i}$};
    \node[nodeHid] (h2\i) at (6.5, 2.0 - \i*1.0) {$h_{2,\i}$};
    \node[nodeOut] (out\i) at (10.0, 2.0 - \i*1.0) {词表 Logits $\hat{y}_\i$};
  }
  
  \foreach \i in {1, 2, 3} {
    \foreach \j in {1, 2, 3} {
      \draw[->, gray!50] (in\i) -- (h1\j);
      \draw[->, gray!50] (h1\i) -- (h2\j);
      \draw[->, gray!50] (h2\i) -- (out\j);
    }
  }
\end{tikzpicture}
\caption{多层感知机（MLP）特征变换与全连接信息流动图}
\end{figure}
"""

TIKZ_TRAIN_LOOP = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=0.92, >=stealth,
  stepBox/.style={rectangle, rounded corners=3pt, draw=primary, fill=primary!6, font=\sffamily\footnotesize, align=center, inner sep=5pt}
]
  \node[stepBox] (s1) at (0, 0) {输入数据 $X$};
  \node[stepBox, fill=secondary!10, draw=secondary] (s2) at (4.0, 0) {前向传播计算\\(网络前向推理)};
  \node[stepBox] (s3) at (8.5, 0) {模型预测值 $\hat{Y}$};
  \node[stepBox, fill=warnred!10, draw=warnred] (s4) at (8.5, -2.0) {计算损失 (Loss)\\与真实标签 $Y$ 对比};
  \node[stepBox, fill=tipgreen!10, draw=tipgreen] (s5) at (4.0, -2.0) {反向传播 (Backward)\\计算各参数梯度 $\nabla_\theta L$};
  \node[stepBox] (s6) at (0, -2.0) {优化器更新参数\\$\theta \leftarrow \theta - \eta \nabla L$};
  
  \draw[->, very thick, primary] (s1) -- (s2);
  \draw[->, very thick, primary] (s2) -- (s3);
  \draw[->, very thick, warnred] (s3) -- (s4);
  \draw[->, very thick, tipgreen] (s4) -- (s5);
  \draw[->, very thick, tipgreen] (s5) -- (s6);
  \draw[->, thick, dashed, primary] (s6) -- (s1) node[midway, left, font=\scriptsize\sffamily] {迭代下一轮};
\end{tikzpicture}
\caption{深度学习模型训练闭环：前向计算、损失衡量与反向梯度传播}
\end{figure}
"""

TIKZ_RNN_UNROLL = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=0.9, >=stealth,
  cell/.style={rectangle, rounded corners=3pt, draw=primary, fill=primary!10, font=\sffamily\footnotesize\bfseries, minimum width=2.4cm, minimum height=1.0cm},
  io/.style={rectangle, rounded corners=2pt, draw=secondary, fill=secondary!8, font=\sffamily\scriptsize}
]
  \node[io] (x1) at (0, 0) {$x_1$ (\texttt{"我"})};
  \node[cell] (rnn1) at (0, 1.8) {RNN 单元};
  \node[io, fill=tipgreen!10, draw=tipgreen] (h1) at (0, 3.6) {隐状态 $h_1$};
  
  \node[io] (x2) at (4.0, 0) {$x_2$ (\texttt{"喜欢"})};
  \node[cell] (rnn2) at (4.0, 1.8) {RNN 单元};
  \node[io, fill=tipgreen!10, draw=tipgreen] (h2) at (4.0, 3.6) {隐状态 $h_2$};
  
  \node[io] (x3) at (8.0, 0) {$x_3$ (\texttt{"大模型"})};
  \node[cell] (rnn3) at (8.0, 1.8) {RNN 单元};
  \node[io, fill=tipgreen!10, draw=tipgreen] (h3) at (8.0, 3.6) {隐状态 $h_3$};
  
  \draw[->, thick] (x1) -- (rnn1);
  \draw[->, thick] (rnn1) -- (h1);
  \draw[->, thick] (x2) -- (rnn2);
  \draw[->, thick] (rnn2) -- (h2);
  \draw[->, thick] (x3) -- (rnn3);
  \draw[->, thick] (rnn3) -- (h3);
  
  \draw[->, very thick, warnred] (rnn1) -- (rnn2) node[midway, above, font=\tiny\sffamily] {时序记忆传递};
  \draw[->, very thick, warnred] (rnn2) -- (rnn3) node[midway, above, font=\tiny\sffamily] {语境累加};
\end{tikzpicture}
\caption{循环神经网络（RNN）沿时序展开的信息流动机制}
\end{figure}
"""

TIKZ_TENSOR_HIERARCHY = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=0.92, >=stealth,
  tensor/.style={rectangle, rounded corners=3pt, align=center, minimum width=2.3cm, minimum height=1.1cm, text centered, font=\sffamily\footnotesize, draw=primary, fill=primary!6}
]
  \node[tensor] (T0) at (0,0) {\textbf{标量 (Scalar)}\\[0.2em]\scriptsize 0 维 $\cdot$ 单个数值};
  \node[tensor] (T1) at (3.2,0) {\textbf{向量 (Vector)}\\[0.2em]\scriptsize 1 维 $\cdot$ 一组有序数值};
  \node[tensor] (T2) at (6.4,0) {\textbf{矩阵 (Matrix)}\\[0.2em]\scriptsize 2 维 $\cdot$ 二维平面网格};
  \node[tensor, fill=tipgreen!10, draw=tipgreen] (T3) at (9.8,0) {\textbf{张量 (Tensor)}\\[0.2em]\scriptsize 3 维及以上 $\cdot$ 多维数据体};
  
  \draw[->, very thick, primary] (T0) -- (T1);
  \draw[->, very thick, primary] (T1) -- (T2);
  \draw[->, very thick, primary] (T2) -- (T3);
\end{tikzpicture}
\caption{数据物理维度的层级跃迁（标量 $\to$ 向量 $\to$ 矩阵 $\to$ 张量）}
\end{figure}
"""

TIKZ_VECTOR_ANGLE = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.2, >=stealth]
  \coordinate (O) at (0,0);
  \coordinate (A) at (3.5,0);
  \coordinate (B) at (2.8,2.2);
  
  \draw[->, very thick, primary] (O) -- (A) node[right, font=\small\sffamily\bfseries] {向量 $\vec{a}$};
  \draw[->, very thick, secondary] (O) -- (B) node[above right, font=\small\sffamily\bfseries] {向量 $\vec{b}$};
  
  \draw[thick, warnred, ->] (0.8,0) arc (0:38:0.8) node[midway, right, font=\small\sffamily] {$\theta$};
  
  \node[font=\footnotesize\sffamily, align=left, color=darktext] at (4.2, 1.2) {
    $\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos\theta$\\
    $\theta \to 0^\circ \implies \cos\theta \to 1$ (最相似)\\
    $\theta = 90^\circ \implies \cos\theta = 0$ (正交无关)
  };
\end{tikzpicture}
\caption{高维向量内积与夹角余弦相似度直观几何解释}
\end{figure}
"""

TIKZ_EXP_CURVE = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.0, >=stealth]
  \draw[->, thick, darktext] (-3.5,0) -- (2.5,0) node[right, font=\small\sffamily] {$x$};
  \draw[->, thick, darktext] (0,-0.3) -- (0,3.8) node[above, font=\small\sffamily] {$y = \exp(x)$};
  
  \draw[domain=-3.2:1.3, smooth, variable=\x, very thick, primary] plot ({\x}, {exp(\x)});
  
  \fill[warnred] (0,1) circle (2pt) node[above left, font=\footnotesize\sffamily] {$(0, 1)$};
  \draw[dashed, codegray] (-3.2, 0) -- (2.2, 0);
  
  \node[font=\footnotesize\sffamily, align=left, draw=boxborder, fill=lightbg, rounded corners=2pt, inner sep=4pt] at (-1.5, 2.2) {
    $e^x > 0$ 恒成立 (严格正定)\\
    单调严格递增\\
    导数 $\frac{d}{dx}e^x = e^x$
  };
\end{tikzpicture}
\caption{$\exp(x) = e^x$ 的严格正定性与单调性}
\end{figure}
"""

TIKZ_QKV_BRANCH = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=0.88, >=stealth,
  box/.style={rectangle, rounded corners=3pt, minimum width=2.6cm, minimum height=0.75cm, text centered, font=\sffamily\footnotesize\bfseries, draw=primary, fill=primary!8}
]
  \node[box, fill=secondary!15, draw=secondary] (X) at (0, 0) {输入向量序列 $X$};
  
  \node[box] (Q) at (4.5, 1.6) {Query ($Q = X W_q$)};
  \node[box] (K) at (4.5, 0) {Key ($K = X W_k$)};
  \node[box] (V) at (4.5, -1.6) {Value ($V = X W_v$)};
  
  \node[anchor=west, font=\sffamily\scriptsize, color=darktext] at (6.2, 1.6) {当前词想要探寻什么信息？};
  \node[anchor=west, font=\sffamily\scriptsize, color=darktext] at (6.2, 0) {当前词有什么特征可被检索？};
  \node[anchor=west, font=\sffamily\scriptsize, color=darktext] at (6.2, -1.6) {当前词承载的实际内容是什么？};
  
  \draw[->, thick, primary] (X.east) -- ++(1.2, 0) |- (Q.west);
  \draw[->, thick, primary] (X.east) -- (K.west);
  \draw[->, thick, primary] (X.east) -- ++(1.2, 0) |- (V.west);
\end{tikzpicture}
\caption{自注意力机制三元组 $Q, K, V$ 投影与语义物理意义}
\end{figure}
"""

TIKZ_ROOFLINE = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.0, >=stealth]
  \draw[->, thick, darktext] (0,0) -- (8.5,0) node[below right, font=\small\sffamily] {算术强度 $I$ (FLOP/Byte)};
  \draw[->, thick, darktext] (0,0) -- (0,4.8) node[above left, font=\small\sffamily] {计算性能 $P$ (FLOP/s)};
  
  \coordinate (O) at (0,0);
  \coordinate (Ridge) at (3.5,3.6);
  \coordinate (MaxPeak) at (8.0,3.6);
  
  \draw[very thick, primary] (O) -- (Ridge) node[midway, above left, sloped, font=\small\sffamily\bfseries, color=primary] {访存受限区 (斜率 = 显存带宽 $\beta$)};
  \draw[very thick, warnred] (Ridge) -- (MaxPeak) node[midway, above, font=\small\sffamily\bfseries, color=warnred] {算力受限区 (峰值算力 $\pi$)};
  
  \draw[dashed, codegray] (3.5,0) node[below, font=\small\sffamily] {$I^* = \frac{\pi}{\beta}$ (拐点 Ridge Point)} -- (Ridge);
  \draw[dashed, codegray] (0,3.6) node[left, font=\small\sffamily] {$\pi$} -- (Ridge);
  
  \fill[warnred] (Ridge) circle (2.5pt);
\end{tikzpicture}
\caption{GPU Roofline 模型与瓶颈判定边界}
\end{figure}
"""

TIKZ_PAGED_ATTN_MEM = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=0.95]
  \node[anchor=west, font=\sffamily\small\bfseries] at (0, 2.5) {传统连续预留 (按 max\_len=2048 预分配，严重浪费)：};
  
  \node[anchor=east, font=\sffamily\footnotesize] at (2.2, 1.8) {请求 A (实际用 300):};
  \draw[fill=primary!70, draw=primary!90] (2.4, 1.5) rectangle (4.2, 2.1) node[midway, white, font=\tiny\bfseries] {已用 300};
  \draw[fill=codegray!15, draw=codegray!60, dashed] (4.2, 1.5) rectangle (11.0, 2.1) node[midway, codegray, font=\footnotesize] {内部碎片 (未用浪费 1748)};
  
  \node[anchor=east, font=\sffamily\footnotesize] at (2.2, 0.8) {请求 B (实际用 1200):};
  \draw[fill=primary!70, draw=primary!90] (2.4, 0.5) rectangle (7.8, 1.1) node[midway, white, font=\tiny\bfseries] {已用 1200};
  \draw[fill=codegray!15, draw=codegray!60, dashed] (7.8, 0.5) rectangle (11.0, 1.1) node[midway, codegray, font=\footnotesize] {内部碎片 (848)};
  
  \draw[<->, thick, darktext] (2.4, 0.1) -- (11.0, 0.1) node[midway, below, font=\scriptsize\sffamily] {连续预留长度 max\_len = 2048 tokens};
\end{tikzpicture}
\caption{传统连续显存分配的内部碎片问题}
\end{figure}
"""

EMOJI_MAP = {
    '👉': '-->',
    '🔬': '[实验]',
    '⏱': '[计时]',
    '⚠️': '[注意]',
    '⚠': '[注意]',
    '🚀': '[起步]',
    '💡': '[思考]',
    '✅': '[OK]',
    '✓': '[OK]',
    '❌': '[X]',
    '✗': '[X]',
    '🌳': '[概览]',
    '🧭': '[指南]',
    '🌟': '[重点]',
    '🏛': '[架构]',
    '⚡': '[加速]',
    '🧠': '[决策]',
    '🏭': '[硬件]',
    '🧊': '[量化]',
    '🗄': '[系统]',
    '🌐': '[并行]',
    '🎲': '[采样]',
    '🧮': '[推导]',
    '🎯': '[目标]',
    '🔍': '[观察]',
    '🟢': '[草稿]',
    '🔵': '[目标]',
    '📖': '[阅读]',
    '📐': '[几何]',
    '🔁': '[反向]',
    '📊': '[图表]',
    '📈': '[增长]',
    '📉': '[下降]',
    '📌': '[重点]',
    '📎': '[附件]',
    '🔧': '[工具]',
    '🛠': '[工具]',
    '🛡': '[防护]',
    '🥇': '[第一阶段]',
    '🥈': '[第二阶段]',
    '🥉': '[第三阶段]',
    '📜': '[文献]',
    '📚': '[资料]',
    '💻': '[代码]',
    '📂': '[目录]',
    '📄': '[文档]',
    '🧱': '[模块]',
    '🪓': '[剖析]',
    '🧩': '[组件]',
    '🧪': '[测试]',
    '🏎': '[高速]',
    '🐢': '[低速]',
    '🏃': '[执行]',
    '🏆': '[检验]',
    '🏋': '[演练]',
    '💥': '[爆炸]',
    '💣': '[隐患]',
    '🌋': '[瓶颈]',
    '🌀': '[循环]',
    '🌓': '[对比]',
    '🌡': '[温度]',
    '🌱': '[启蒙]',
    '🧑': '[人员]',
    '🧈': '[基础]',
    '🛤': '[路径]',
    '🚧': '[施工]',
    '🚫': '[禁用]',
    '🔗': '[链接]',
    '🔢': '[数字]'
}

def clean_emojis_and_symbols(text):
    """将 Emoji 字符转换为清晰文本标识"""
    for k, v in EMOJI_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]', '', text)
    text = text.replace('\ufe0f', '').replace('\ufe0e', '')
    text = text.replace('\u200d', '')
    text = text.replace('ℓ', 'l')
    text = text.replace('①', '(1)').replace('②', '(2)').replace('③', '(3)').replace('④', '(4)').replace('⑤', '(5)')
    return text

def clean_title_for_latex(title):
    """清洗标题文本用于 LaTeX 章节命令"""
    title = clean_emojis_and_symbols(title)
    title = re.sub(r'^第\s*\d+\s*篇[（(][^）)]+[）)]\s*[：:]\s*', '', title)
    title = re.sub(r'^第\s*\d+\s*篇\s*[：:]\s*', '', title)
    title = re.sub(r'^[一二三四五六七八九十]+[、.\s]\s*', '', title)
    title = re.sub(r'^\d+[、.\s]\s*', '', title)
    # 处理粗体与代码标记
    title = re.sub(r'\*\*([^*]+)\*\*', r'\1', title)
    title = title.replace('*', '')
    title = title.replace('`', '')
    title = title.replace('\\', '')
    title = title.replace('_', r'\_')
    title = title.replace('%', r'\%')
    title = title.replace('&', r'\&')
    title = title.replace('#', r'\#')
    
    # 替换标题中的数学符号，兼顾 LaTeX 排版与 PDF 书签
    title_math_map = [
        ('≈', r'\texorpdfstring{$\approx$}{≈}'),
        ('≠', r'\texorpdfstring{$\neq$}{!=}'),
        ('→', r'\texorpdfstring{$\rightarrow$}{->}'),
        ('←', r'\texorpdfstring{$\leftarrow$}{<-}'),
        ('×', r'\texorpdfstring{$\times$}{x}'),
        ('≤', r'\texorpdfstring{$\le$}{<=}'),
        ('≥', r'\texorpdfstring{$\ge$}{>=}'),
        ('π', r'\texorpdfstring{$\pi$}{pi}'),
        ('α', r'\texorpdfstring{$\alpha$}{alpha}'),
        ('β', r'\texorpdfstring{$\beta$}{beta}'),
        ('~', r'\textasciitilde{}'),
    ]
    for char, esc in title_math_map:
        title = title.replace(char, esc)
        
    return title.strip()

def clean_text_for_latex(text):
    """转义普通正文中的 LaTeX 保留符号并安全映射数学符号（单遍安全替换）"""
    text = clean_emojis_and_symbols(text)
    
    # 第一步：单遍替换 LaTeX 保留符号，防止级联重入（如 \ 产生 \textbackslash{} 后内部 {} 被二次替换）
    def repl_esc(m):
        ch = m.group(0)
        return {
            '\\': r'\textbackslash{}',
            '%': r'\%',
            '&': r'\&',
            '#': r'\#',
            '_': r'\_',
            '^': r'\^{}',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}'
        }[ch]
    text = re.sub(r'([\\%&#_\^{}~])', repl_esc, text)
    
    # 第二步：转义数学符号、希腊字母与几何图元
    math_symbols = [
        ('→', r'$\rightarrow$'),
        ('←', r'$\leftarrow$'),
        ('⇒', r'$\Rightarrow$'),
        ('↔', r'$\leftrightarrow$'),
        ('≈', r'$\approx$'),
        ('≠', r'$\neq$'),
        ('≤', r'$\le$'),
        ('≥', r'$\ge$'),
        ('×', r'$\times$'),
        ('÷', r'$\div$'),
        ('·', r'$\cdot$'),
        ('…', r'\dots{}'),
        ('✓', r'\checkmark{}'),
        ('✔', r'\checkmark{}'),
        ('❌', r'$\times$'),
        ('✗', r'$\times$'),
        ('±', r'$\pm$'),
        ('°', r'$^\circ$'),
        ('√', r'$\sqrt{}$'),
        ('∞', r'$\infty$'),
        ('⭐', r'$\star$'),
        ('α', r'$\alpha$'),
        ('β', r'$\beta$'),
        ('γ', r'$\gamma$'),
        ('δ', r'$\delta$'),
        ('ε', r'$\varepsilon$'),
        ('θ', r'$\theta$'),
        ('λ', r'$\lambda$'),
        ('μ', r'$\mu$'),
        ('π', r'$\pi$'),
        ('σ', r'$\sigma$'),
        ('τ', r'$\tau$'),
        ('ω', r'$\omega$'),
        ('ℓ', r'$\ell$'),
        ('ᵀ', r'$^T$'),
        ('──►', r'$\longrightarrow$'),
        ('──', r'——'),
        ('►', r'$\blacktriangleright$'),
        ('▲', r'$\blacktriangle$'),
        ('▼', r'$\blacktriangledown$'),
        ('■', r'$\blacksquare$'),
        ('□', r'$\square$')
    ]
    for char, esc in math_symbols:
        text = text.replace(char, esc)
        
    # 第三步：防止行首 '[' 被 LaTeX 当作可选参数
    text = re.sub(r'^(\s*)\[', r'\1{}[', text)
    
    return text

def convert_markdown_to_latex(content, is_chapter=True):
    """核心 Markdown 到 LaTeX 解析器（采用全字母安全占位符）"""
    content = content.replace('\r\n', '\n')
    
    code_blocks = []
    display_math_blocks = []
    inline_math_blocks = []
    inline_code_blocks = []
    table_blocks = []
    custom_blocks = []

    def save_custom_block(latex_str):
        idx = len(custom_blocks)
        custom_blocks.append(latex_str)
        return f"\n\nXXXCUSTOMBLOCKXXX{idx}XXX\n\n"
    
    # 0. 将各种伪代码/ASCII 图形替换为高质量 TikZ 与 LaTeX 结构
    content = re.sub(r'```\s*输入特征 x1.*?```', r'\n\nXXXTIKZPERCEPTRONXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*输入层 \(Input Layer\).*?```', r'\n\nXXXTIKZMLPLAYERSXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*【前向传播 Forward】.*?```', r'\n\nXXXTIKZTRAINLOOPXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*输入词 x_1.*?```', r'\n\nXXXTIKZRNNUNROLLXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*标量 \(0维\).*?```', r'\n\nXXXTIKZTENSORHIERARCHYXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*b\s*\n\s*/\s*\n\s*/\s*夹角 θ.*?```', r'\n\nXXXTIKZVECTORANGLEXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*y = e\^x 的曲线.*?```', r'\n\nXXXTIKZEXPCURVEXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*输入文本向量 X.*?```', r'\n\nXXXTIKZQKVBRANCHXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*性能 \(FLOP/s\).*?```', r'\n\nXXXTIKZROOFLINEXXX\n\n', content, flags=re.DOTALL)
    content = re.sub(r'```\s*请求 A \(实际用 300\).*?```', r'\n\nXXXTIKZPAGEDATTNMEMXXX\n\n', content, flags=re.DOTALL)

    # 自回归解码步数表格
    auto_reg_table = r"""\begin{center}
\small
\begin{tabularx}{\textwidth}{p{1.8cm} X l}
\toprule
\textbf{解码步数} & \textbf{模型前向输入上下文（Context Tokens）} & \textbf{预测输出下一个 Token} \\
\midrule
第 1 步 & \texttt{["北京", "的", "特产", "是", "什么", "？"]} & \textbf{"烤"} \\
第 2 步 & \texttt{["北京", "的", "特产", "是", "什么", "？", "烤"]} & \textbf{"鸭"} \\
第 3 步 & \texttt{["北京", "的", "特产", "是", "什么", "？", "烤", "鸭"]} & \textbf{"。"}（触发 EOS 停止） \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*Step 1: \["北京".*?```', lambda m: save_custom_block(auto_reg_table), content, flags=re.DOTALL)

    # Prefill vs Decode 对比表格
    prefill_decode_table = r"""\begin{center}
\small
\begin{tabularx}{\textwidth}{p{3.0cm} X X}
\toprule
\textbf{特征维度} & \textbf{阶段 1：Prefill 阶段 (首字计算)} & \textbf{阶段 2：Decode 阶段 (增量生成)} \\
\midrule
\textbf{输入形态} & 一次性输入全部 Prompt (如 1000 字) & 每次仅输入最新生成的 \textbf{1 个 Token} \\
\textbf{计算模式} & 大规模矩阵乘法 (GEMM) & 极小规模矩阵-向量乘法 (GEMV) \\
\textbf{硬件瓶颈} & \textbf{算力受限 (Compute-Bound)}，核心满载 & \textbf{访存受限 (Memory-Bound)}，带宽跑满 \\
\textbf{核心考核指标} & \textbf{TTFT} (首字时间，用户等待延迟) & \textbf{TPOT} (每字时间，流式打字流畅度) \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*用户输入 Prompt:.*?Prefill 阶段.*?```', lambda m: save_custom_block(prefill_decode_table), content, flags=re.DOTALL)

    # 第 t 步推理 KV Cache 增量
    kv_step_box = r"""\begin{mathBox}[第 $t$ 步推理：KV Cache 增量运算过程]
\begin{enumerate}
  \item \textbf{单向量投影}：仅对最新生成的单个 Token 向量 $x_t \in \mathbb{R}^{1 \times d_{\mathrm{model}}}$ 计算投影：
  \[
  q_t = x_t W_q, \quad k_t = x_t W_k, \quad v_t = x_t W_v
  \]
  \item \textbf{更新增量缓存}：将新产生的单步 $k_t, v_t$ 追加到已有显存缓存中：
  \[
  K_{\mathrm{all}} = \begin{bmatrix} K_{\mathrm{cache}} \\ k_t \end{bmatrix}, \quad V_{\mathrm{all}} = \begin{bmatrix} V_{\mathrm{cache}} \\ v_t \end{bmatrix}
  \]
  \item \textbf{注意力打分}：单向量 $q_t$ 与整个长序列 $K_{\mathrm{all}}$ 做矩阵乘法，得到 $[1, t]$ 注意力分布，再加权 $V_{\mathrm{all}}$ 得出输出向量！
\end{enumerate}
\end{mathBox}"""
    content = re.sub(r'```\s*【第 t 步推理】.*?```', lambda m: save_custom_block(kv_step_box), content, flags=re.DOTALL)

    # Logits 打分表格
    logits_table = r"""\begin{center}
\small
\begin{tabularx}{0.7\textwidth}{l c X}
\toprule
\textbf{候选 Token} & \textbf{模型输出 Logit 打分} & \textbf{选择策略判定} \\
\midrule
\texttt{"的"} & $14.5$ & \textbf{最高分！Greedy 策略直接选中} \\
\texttt{"是"} & $12.1$ & 次高分 \\
\texttt{"在"} & $9.3$ & 低分 \\
\texttt{"飞船"} & $-2.0$ & 极低概率 \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*词表打分 Logits:.*?```', lambda m: save_custom_block(logits_table), content, flags=re.DOTALL)

    # Top-K 截断
    topk_table = r"""\begin{center}
\small
\begin{tabularx}{\textwidth}{X c X}
\toprule
\textbf{高概率候选词（保留采样池）} & \textbf{截断阈值} & \textbf{长尾低概率词（强制置 $-\infty$）} \\
\midrule
前 $K$ 个最高概率词: \texttt{[词 1, 词 2, $\dots$, 词 $K$]} & $\longleftrightarrow$ & 剩余词: \texttt{[词 $K+1$, $\dots$, 词 $V$]} 全部过滤 \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*全部 15 万个词.*?```', lambda m: save_custom_block(topk_table), content, flags=re.DOTALL)

    # Top-P 动态截断
    topp_table = r"""\begin{center}
\small
\begin{tabularx}{\textwidth}{p{3.5cm} c X}
\toprule
\textbf{语境场景} & \textbf{候选 Token 概率累积} & \textbf{动态候选池判定} \\
\midrule
\textbf{极确定语境} (如代码/事实) & \texttt{"AI"}: 92\% $\ge 90\%$ & \textbf{立即截断！候选池仅保留 1 个词} \\
\midrule
\textbf{发散创意语境} (如诗歌/创作) & \texttt{"落日"}: 25\%, \texttt{"晚霞"}: 20\%, $\dots$ & \textbf{累积至 90.5\% 截断！动态保留 15 个候选词} \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*候选词按概率从高到低排序:.*?```', lambda m: save_custom_block(topp_table), content, flags=re.DOTALL)

    # 学习路线
    path_list = r"""\begin{enumerate}
  \item 第 1 篇：计算机如何从规则、统计到被 Transformer 彻底征服
  \item 第 2 篇：向量空间几何、点积的几何意义与 Safe Softmax 溢出防护
  \item 第 3 篇：导数、反向传播、交叉熵，Softmax 雅可比与 $p - y$ 梯度
  \item 第 4 篇：严格证明 Attention 为什么必须除以 $\sqrt{d_k}$，推导 RoPE 相对位置内积恒等性
  \item 第 5 篇：从原版 Transformer 到 Llama / DeepSeek，每一处改动的历史动机
  \item 第 6 篇：推理与训练的本质区别，代数展开 KV Cache 消除冗余并剖析显存挑战
  \item 第 7 篇：温度参数在 $T \to 0$ 和 $T \to \infty$ 下的数学极限，Top-K、Top-P 与重复惩罚
\end{enumerate}"""
    content = re.sub(r'```\s*第 1 篇：计算机如何从规则.*?```', lambda m: save_custom_block(path_list), content, flags=re.DOTALL)

    # 算子融合与访存
    fusion_table = r"""\begin{center}
\small
\begin{tabularx}{\textwidth}{p{2.8cm} X}
\toprule
\textbf{执行模式} & \textbf{显存（HBM）与片上缓存（SRAM）数据流动过程} \\
\midrule
\textbf{朴素未融合算子} & 读 $x$ (HBM) $\to$ RMSNorm $\to$ 写回 HBM $\to$ 读回 $\to$ 乘 $\gamma$ $\to$ 写回 $\to$ 读回 $\to$ 残差加法 $\to$ 写回 HBM（\textbf{访存往返 6 次！}） \\
\midrule
\textbf{Kernel Fusion 融合} & 读 $x$ (HBM) $\to$ \textbf{片上 SRAM / 寄存器一次性完成计算} $\to$ 写回 HBM（\textbf{访存仅 1 次，打满带宽！}） \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*读 x \(HBM\).*?```', lambda m: save_custom_block(fusion_table), content, flags=re.DOTALL)

    # FlashAttention 朴素访存对比
    flash_io_table = r"""\begin{center}
\small
\begin{tabularx}{\textwidth}{l X X}
\toprule
\textbf{标准 Attention 步骤} & \textbf{数学算子公式} & \textbf{朴素显存读写量（HBM IO）} \\
\midrule
1. 计算打分矩阵 & $S = Q K^T / \sqrt{d}$ & 读 $Q, K$ ($2Nd$) $\longrightarrow$ 写 $S$ 到显存 ($N^2$) \\
2. 概率归一化 & $P = \mathrm{softmax}(S)$ & 读 $S$ ($N^2$) $\longrightarrow$ 写 $P$ 到显存 ($N^2$) \\
3. 加权求和输出 & $O = P V$ & 读 $P$ ($N^2$), $V$ ($Nd$) $\longrightarrow$ 写 $O$ 到显存 ($Nd$) \\
\midrule
\textbf{总访存量} & \multicolumn{2}{l}{\textbf{高达 $4Nd + 2N^2$ 字节（受 $O(N^2)$ 显存访问严重瓶颈束缚）}} \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*S = Q K.*?```', lambda m: save_custom_block(flash_io_table), content, flags=re.DOTALL)

    # FlashAttention 分块伪代码 -> 标上 python
    content = re.sub(r'```\s*(把 Q 按行切成若干块.*?```)', r'```python\n\1', content, flags=re.DOTALL)

    # PagedAttention 块表映射
    paged_table = r"""\begin{center}
\small
\begin{tabularx}{\textwidth}{l X}
\toprule
\textbf{请求 ID} & \textbf{逻辑块到物理块映射表 (Block Table)} \\
\midrule
\textbf{请求 A} & 逻辑块 0 $\to$ 物理块 7, \quad 逻辑块 1 $\to$ 物理块 2, \quad 逻辑块 2 $\to$ 物理块 9 \\
\textbf{请求 B} & 逻辑块 0 $\to$ 物理块 4, \quad 逻辑块 1 $\to$ 物理块 0, \quad 逻辑块 2 $\to$ 物理块 5 \\
\midrule
\textbf{核心优势} & \textbf{按需分配固定大小的物理块 (如 16 tokens)，物理显存无需连续，内部碎片近乎为零！} \\
\bottomrule
\end{tabularx}
\end{center}"""
    content = re.sub(r'```\s*请求 A 的块表:.*?```', lambda m: save_custom_block(paged_table), content, flags=re.DOTALL)

    # 1. 替换 Mermaid 为 TikZ
    content = re.sub(r'```mermaid\s+flowchart TD\s+P\["1957.*?```', r'XXXTIKZAITREEXXX', content, flags=re.DOTALL)
    content = re.sub(r'```mermaid\s+flowchart TD\s+S0\["原始文本.*?```', r'XXXTIKZTRANSFORMERPIPELINEXXX', content, flags=re.DOTALL)
    content = re.sub(r'```mermaid\s+sequenceDiagram.*?```', r'XXXTIKZSPECULATIVEDECODINGXXX', content, flags=re.DOTALL)
    content = re.sub(r'```mermaid.*?```', '', content, flags=re.DOTALL)
    
    # 2. 提取代码块
    def save_code_block(match):
        idx = len(code_blocks)
        lang = match.group(1).strip()
        code = match.group(2)
        code_blocks.append((lang, code))
        return f"\n\nXXXCODEBLOCKXXX{idx}XXX\n\n"
    content = re.sub(r'```([^\n]*)\n(.*?)```', save_code_block, content, flags=re.DOTALL)
    
    # 3. 提取 Display Math
    def save_display_math(match):
        idx = len(display_math_blocks)
        math = match.group(1).strip()
        math = math.replace('！', '!').replace('？', '?')
        display_math_blocks.append(math)
        return f"\n\nXXXMATHBLOCKXXX{idx}XXX\n\n"
    content = re.sub(r'\$\$(.+?)\$\$', save_display_math, content, flags=re.DOTALL)
    
    # 4. 提取 Inline Math
    def save_inline_math(match):
        idx = len(inline_math_blocks)
        math = match.group(1)
        math = math.replace('！', '!').replace('？', '?')
        inline_math_blocks.append(math)
        return f"XXXMATHINLINEXXX{idx}XXX"
    content = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save_inline_math, content)
    
    # 4.5 提取 Inline Code（在普通文本转义之前提取，彻底杜绝双重转义）
    def save_inline_code(match):
        idx = len(inline_code_blocks)
        c = match.group(1)
        escaped = ""
        for ch in c:
            if ch == '\\': escaped += r'\textbackslash{}'
            elif ch == '_': escaped += r'\_'
            elif ch == '%': escaped += r'\%'
            elif ch == '&': escaped += r'\&'
            elif ch == '#': escaped += r'\#'
            elif ch == '{': escaped += r'\{'
            elif ch == '}': escaped += r'\}'
            elif ch == '^': escaped += r'\^{}'
            elif ch == '~': escaped += r'\textasciitilde{}'
            elif ch == '<': escaped += r'$<$'
            elif ch == '>': escaped += r'$>$'
            else: escaped += ch
        inline_code_blocks.append(r'\texttt{' + escaped + '}')
        return f"XXXINLINECODEXXX{idx}XXX"
    content = re.sub(r'`([^`\n]+)`', save_inline_code, content)
    
    # 5. 提取 Markdown 表格（支持缩进表格）
    def save_table(match):
        tbl = match.group(0).strip()
        tbl_lines = [l.strip() for l in tbl.split('\n') if l.strip()]
        if len(tbl_lines) >= 2 and any('---' in l for l in tbl_lines[:3]):
            idx = len(table_blocks)
            table_blocks.append(tbl)
            return f"\n\nXXXTABLEXXX{idx}XXX\n\n"
        return match.group(0)
    content = re.sub(r'((?:^[ \t]*\|[^\n]+\|\n?)+)', save_table, content, flags=re.MULTILINE)
    
    # 6. 内联文本格式转换
    def convert_inlines(line):
        line = re.sub(r'\*\*([^\*]+)\*\*', r'\\textbf{\1}', line)
        line = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'\\textit{\1}', line)
        
        def repl_link(m):
            txt = m.group(1)
            url = m.group(2)
            if url.startswith('http://') or url.startswith('https://'):
                return r'\href{' + url + '}{' + txt + '}'
            else:
                return r'\textbf{' + txt + '}'
        line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', repl_link, line)
        return line

    lines = content.split('\n')
    output_lines = []
    
    in_quote = False
    quote_lines = []
    quote_type = "noteBox"
    quote_title = "核心导读与背景"
    in_list = None
    
    def flush_quote():
        nonlocal in_quote, quote_lines, quote_type, quote_title
        if not in_quote:
            return
        body = "\n".join(quote_lines).strip()
        body_tex = convert_inlines(body)
        output_lines.append(f"\\begin{{{quote_type}}}[{quote_title}]\n{body_tex}\n\\end{{{quote_type}}}\n")
        in_quote = False
        quote_lines = []
        quote_type = "noteBox"
        quote_title = "核心导读与背景"
        
    def flush_list():
        nonlocal in_list
        if in_list is not None:
            output_lines.append(f"\\end{{{in_list}}}\n")
            in_list = None

    for line in lines:
        stripped = line.strip()
        
        # 引用框处理
        if stripped.startswith('>'):
            flush_list()
            q_text = re.sub(r'^>\s*', '', stripped)
            if not in_quote:
                in_quote = True
                if any(k in q_text for k in ['💡', '思考', '直觉', '启发', '动机']):
                    quote_type = "tipBox"
                    quote_title = "核心直觉与思考"
                elif any(k in q_text for k in ['⚠️', '警告', '注意', '避坑', '坑']):
                    quote_type = "warningBox"
                    quote_title = "避坑指南与核心警示"
                elif any(k in q_text for k in ['定理', '推导', '证明', '公式']):
                    quote_type = "mathBox"
                    quote_title = "核心数学定理推导"
                else:
                    quote_type = "noteBox"
                    quote_title = "核心导读与背景"
            
            q_text = re.sub(r'^\*\*?(前言|阅读前须知|核心思考|思考|注意|提示|避坑指南)[：:]\*\*?\s*', '', q_text)
            q_text = clean_text_for_latex(q_text)
            quote_lines.append(q_text)
            continue
        else:
            if in_quote:
                flush_quote()
                
        # 分割线
        if re.match(r'^(?:---|\*\*\*|___)\s*$', stripped):
            flush_list()
            output_lines.append("\n\\vspace{0.8em}\\hrule\\vspace{0.8em}\n")
            continue
            
        # 标题处理
        if stripped.startswith('#'):
            flush_list()
            h_match = re.match(r'^(#+)\s+(.+)$', stripped)
            if h_match:
                level = len(h_match.group(1))
                title = clean_title_for_latex(h_match.group(2))
                
                if level == 1:
                    if is_chapter:
                        output_lines.append(f"\\chapter{{{title}}}\n")
                    else:
                        output_lines.append(f"\\section{{{title}}}\n")
                elif level == 2:
                    output_lines.append(f"\\section{{{title}}}\n")
                elif level == 3:
                    output_lines.append(f"\\subsection{{{title}}}\n")
                else:
                    output_lines.append(f"\\subsubsection{{{title}}}\n")
                continue
                
        # 列表处理
        unordered_match = re.match(r'^[*-]\s+(.+)$', stripped)
        ordered_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        
        if unordered_match:
            if in_list != 'itemize':
                flush_list()
                output_lines.append("\\begin{itemize}")
                in_list = 'itemize'
            item_text = convert_inlines(clean_text_for_latex(unordered_match.group(1)))
            output_lines.append(f"  \\item {item_text}")
            continue
        elif ordered_match:
            if in_list != 'enumerate':
                flush_list()
                output_lines.append("\\begin{enumerate}")
                in_list = 'enumerate'
            item_text = convert_inlines(clean_text_for_latex(ordered_match.group(1)))
            output_lines.append(f"  \\item {item_text}")
            continue
        else:
            flush_list()
            
        # 普通正文
        if stripped:
            conv_line = convert_inlines(clean_text_for_latex(line))
            output_lines.append(conv_line)
        else:
            output_lines.append("")
            
    flush_quote()
    flush_list()
    result = "\n".join(output_lines)
    
    # 7. 还原表格 (使用 center + tabularx)
    for idx, tbl_str in enumerate(table_blocks):
        tbl_lines = [l.strip() for l in tbl_str.split('\n') if l.strip()]
        if len(tbl_lines) < 2:
            tex_table = ""
        else:
            headers = [c.strip() for c in tbl_lines[0].split('|')[1:-1]]
            n_cols = len(headers)
            if n_cols == 2:
                col_spec = "p{3.5cm} X"
            elif n_cols == 3:
                col_spec = "p{3cm} p{4.5cm} X"
            elif n_cols == 4:
                col_spec = "p{2.5cm} p{3.5cm} X X"
            else:
                col_spec = "l " + "X " * max(1, n_cols - 1)
                
            tex_rows = [
                r"\begin{center}",
                r"\small",
                f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}",
                r"\toprule"
            ]
            hdr_cells = ["{" + convert_inlines(clean_text_for_latex(h)) + "}" for h in headers]
            tex_rows.append(" & ".join([r"\textbf" + h for h in hdr_cells]) + r" \\")
            tex_rows.append(r"\midrule")
            
            for row in tbl_lines[2:]:
                cells = [c.strip() for c in row.split('|')[1:-1]]
                while len(cells) < n_cols:
                    cells.append("")
                cells = cells[:n_cols]
                converted_cells = []
                for c in cells:
                    c_tex = convert_inlines(clean_text_for_latex(c))
                    c_tex = c_tex.replace('<br/>', r'\newline ').replace('<br>', r'\newline ')
                    if c_tex.startswith('['):
                        c_tex = '{' + c_tex + '}'
                    converted_cells.append(c_tex)
                tex_rows.append(" & ".join(converted_cells) + r" \\")
                
            tex_rows.append(r"\bottomrule")
            tex_rows.append(r"\end{tabularx}")
            tex_rows.append(r"\end{center}")
            tex_table = "\n".join(tex_rows)
            
        result = result.replace(f"XXXTABLEXXX{idx}XXX", tex_table)
        
    # 8. 还原 TikZ 图表
    result = result.replace("XXXTIKZAITREEXXX", TIKZ_AI_TREE)
    result = result.replace("XXXTIKZTRANSFORMERPIPELINEXXX", TIKZ_TRANSFORMER_PIPELINE)
    result = result.replace("XXXTIKZSPECULATIVEDECODINGXXX", TIKZ_SPECULATIVE_DECODING)
    result = result.replace("XXXTIKZPERCEPTRONXXX", TIKZ_PERCEPTRON)
    result = result.replace("XXXTIKZMLPLAYERSXXX", TIKZ_MLP_LAYERS)
    result = result.replace("XXXTIKZTRAINLOOPXXX", TIKZ_TRAIN_LOOP)
    result = result.replace("XXXTIKZRNNUNROLLXXX", TIKZ_RNN_UNROLL)
    result = result.replace("XXXTIKZTENSORHIERARCHYXXX", TIKZ_TENSOR_HIERARCHY)
    result = result.replace("XXXTIKZVECTORANGLEXXX", TIKZ_VECTOR_ANGLE)
    result = result.replace("XXXTIKZEXPCURVEXXX", TIKZ_EXP_CURVE)
    result = result.replace("XXXTIKZQKVBRANCHXXX", TIKZ_QKV_BRANCH)
    result = result.replace("XXXTIKZROOFLINEXXX", TIKZ_ROOFLINE)
    result = result.replace("XXXTIKZPAGEDATTNMEMXXX", TIKZ_PAGED_ATTN_MEM)
    
    # 8.5 还原自定义结构块 (Custom Blocks)
    for idx, block in enumerate(custom_blocks):
        result = result.replace(f"XXXCUSTOMBLOCKXXX{idx}XXX", block)
    
    # 9. 还原 Display Math
    for idx, math in enumerate(display_math_blocks):
        math_tex = f"\\[\n{math}\n\\]"
        result = result.replace(f"XXXMATHBLOCKXXX{idx}XXX", math_tex)
        
    # 10. 还原 Inline Math
    for idx, math in enumerate(inline_math_blocks):
        math_tex = f"${math}$"
        result = result.replace(f"XXXMATHINLINEXXX{idx}XXX", math_tex)
        
    # 10.5 还原 Inline Code
    for idx, code_tex in enumerate(inline_code_blocks):
        result = result.replace(f"XXXINLINECODEXXX{idx}XXX", code_tex)
        
    # 11. 还原章节内代码块 (使用 lstlisting，frame=leftline)
    for idx, (lang, code) in enumerate(code_blocks):
        code_cleaned = clean_emojis_and_symbols(code.strip('\n'))
        if lang.lower() in ['py', 'python']:
            lst_tag = "[language=Python]"
        elif lang.lower() in ['bash', 'sh', 'shell']:
            lst_tag = "[language=bash]"
        elif lang.lower() in ['c', 'cpp', 'cuda', 'cu']:
            lst_tag = "[language=C++]"
        else:
            lst_tag = "[numbers=none]"
            
        code_tex = f"\\begin{{lstlisting}}{lst_tag}\n{code_cleaned}\n\\end{{lstlisting}}\n"
        result = result.replace(f"XXXCODEBLOCKXXX{idx}XXX", code_tex)
        
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

# ------------------------------------------------------------------------------
# 批处理第一部分：原理与数学推导篇 (zero_to_hero_tutorial)
# ------------------------------------------------------------------------------
def generate_part1():
    print(">>> 正在生成第一部分：理论与数学推导篇...")
    tutorial_files = sorted(glob.glob(os.path.join(REPO_ROOT, "zero_to_hero_tutorial", "*.md")))
    generated_chapters = []
    
    for idx, fpath in enumerate(tutorial_files):
        fname = os.path.basename(fpath)
        print(f"  处理教程 [{idx+1}/{len(tutorial_files)}]: {fname}")
        with open(fpath, 'r', encoding='utf-8') as fp:
            content = fp.read()
            
        latex_content = convert_markdown_to_latex(content, is_chapter=True)
        out_name = f"ch{idx:02d}.tex"
        out_path = os.path.join(CHAPTERS_DIR, out_name)
        with open(out_path, 'w', encoding='utf-8') as fp:
            fp.write(latex_content)
        generated_chapters.append(out_name)
        
    return generated_chapters

# ------------------------------------------------------------------------------
# 批处理第二部分：工业实战与实验源码精解篇 (lab03 - lab13, capstone)
# ------------------------------------------------------------------------------
def generate_part2():
    print(">>> 正在生成第二部分：工业实战与实验源码精解篇...")
    
    lab_definitions = [
        ("lab03_autograd_from_scratch", "从零手写自动微分与反向传播引擎 (Micrograd)", ["micrograd_from_scratch.py"]),
        ("lab04_transformer_microscope", "Transformer 推理显微镜：逐层张量变化与 RoPE 实测", ["01_tensor_trace_step_by_step.py", "02_rope_and_attention_microscope.py"]),
        ("lab05_train_tiny_llama", "从零训练与推理最小完整 Llama 模型", ["model.py", "train.py", "generate.py"]),
        ("lab06_kv_cache", "KV Cache 性能基准测试与吞吐加速比实测", ["kv_cache_benchmark.py"]),
        ("lab07a_sampling", "工业级文本采样策略实操 (Top-K / Top-P / Min-P)", ["sampling_strategies.py"]),
        ("lab07b_qwen_from_scratch", "纯 Python 零依赖 Qwen 模型真实权重推理", ["qwen_from_scratch.py"]),
        ("lab08_memory_and_roofline", "显存占用严谨核算与 GPU 算力 Roofline 实测", ["memory_calculator.py", "measure_gpu_roofline.py"]),
        ("lab09_flash_attention", "FlashAttention 算法仿真与 CUDA Softmax 算子", ["flash_attention_numpy.py"]),
        ("lab10_speculative_decoding", "投机解码仿真实现与无偏性实证校验", ["toy_speculative_decoding.py", "verify_unbiasedness.py"]),
        ("lab11_quantization", "工业量化机制（RTN / SmoothQuant / AWQ / GPTQ）实战", ["quant_basics.py", "advanced_quant_numpy.py"]),
        ("lab12a_continuous_batching", "Continuous Batching（持续批处理）动态调度仿真", ["continuous_batching_sim.py"]),
        ("lab12b_paged_attention", "PagedAttention（分页显存分配机制）仿真", ["paged_kv_cache_sim.py"]),
        ("lab13_parallelism", "多卡张量并行（Tensor Parallelism）切分与 All-Reduce 仿真", ["tensor_parallel_sim.py"]),
        ("capstone_engines_practice", "生产级推理引擎（vLLM / SGLang / TRT-LLM）压测实践", ["bench_openai_server.py"])
    ]
    
    generated_labs = []
    for idx, (lab_dir, lab_title, py_files) in enumerate(lab_definitions):
        print(f"  处理实验 [{idx+1}/{len(lab_definitions)}]: {lab_dir}")
        lab_path = os.path.join(REPO_ROOT, lab_dir)
        readme_path = os.path.join(lab_path, "README.md")
        
        lab_tex = []
        safe_lab_title = clean_title_for_latex(lab_title)
        lab_tex.append(f"\\chapter{{{safe_lab_title}}}\n")
        
        # 1. 包含 README.md
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as fp:
                readme_content = fp.read()
            readme_tex = convert_markdown_to_latex(readme_content, is_chapter=False)
            lab_tex.append(readme_tex)
            
        # 2. 包含关联核心 Python 源码
        lab_tex.append("\n\\section{实验核心源码精选与解析}\n")
        lab_tex.append("本实验配套有完整可运行的工业级验证代码，重点实现细节如下：\n\n")
        
        for py_file in py_files:
            py_path = os.path.join(lab_path, py_file)
            if os.path.exists(py_path):
                with open(py_path, 'r', encoding='utf-8') as fp:
                    code_text = fp.read()
                clean_code = clean_emojis_and_symbols(code_text)
                dest_code_name = f"{lab_dir}_{py_file}"
                dest_code_path = os.path.join(CODE_DIR, dest_code_name)
                with open(dest_code_path, 'w', encoding='utf-8') as cfp:
                    cfp.write(clean_code)
                    
                safe_file_title = py_file.replace('_', r'\_')
                lab_tex.append(f"\\subsection{{源码实现：\\texttt{{{safe_file_title}}}}}\n")
                lab_tex.append(f"\\lstinputlisting[language=Python, caption={{\\texttt{{{safe_file_title}}}}}]{{code/{dest_code_name}}}\n\n")
                
        out_name = f"lab{idx:02d}.tex"
        out_path = os.path.join(LABS_DIR, out_name)
        with open(out_path, 'w', encoding='utf-8') as fp:
            fp.write("\n".join(lab_tex))
        generated_labs.append(out_name)
        
    return generated_labs

# ------------------------------------------------------------------------------
# 生成前言与学习路径
# ------------------------------------------------------------------------------
def generate_preface():
    print(">>> 正在生成前言与学习路径指引...")
    readme_path = os.path.join(REPO_ROOT, "README.md")
    
    preface_tex = []
    preface_tex.append(r"\chapter*{前言：大模型推理的底层物理规律}")
    preface_tex.append(r"\addcontentsline{toc}{chapter}{前言：大模型推理的底层物理规律}")
    
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as fp:
            content = fp.read()
        content = re.sub(r'```mermaid.*?```', '', content, flags=re.DOTALL)
        tex = convert_markdown_to_latex(content, is_chapter=False)
        preface_tex.append(tex)
        
    out_path = os.path.join(CHAPTERS_DIR, "preface.tex")
    with open(out_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(preface_tex))
    return "preface.tex"

# ------------------------------------------------------------------------------
# 主函数：组装 main.tex
# ------------------------------------------------------------------------------
def main():
    preface_file = generate_preface()
    part1_files = generate_part1()
    part2_files = generate_part2()
    
    print(">>> 正在生成主控文件 main.tex...")
    main_tex = [
        r"\documentclass[11pt,a4paper,openany,UTF8]{ctexbook}",
        r"\input{preamble.tex}",
        r"",
        r"\begin{document}",
        r"",
        r"% ----------------------------------------------------------------------",
        r"% 封面设计",
        r"% ----------------------------------------------------------------------",
        r"\begin{titlepage}",
        r"\centering",
        r"\vspace*{2.5cm}",
        r"{\Huge\bfseries\sffamily\color{primary} 大模型推理系统与核心原理}\\[0.6cm]",
        r"{\LARGE\sffamily\color{secondary} 从零基础数学推导到现代工业工程落地}\\[1.2cm]",
        r"{\large\sffamily\color{darktext} 深入自注意力、KV Cache、Roofline 模型、FlashAttention、量化与分布式并行}\\[2.5cm]",
        r"",
        r"\begin{tcolorbox}[enhanced, width=0.88\textwidth, colback=primary!5!white, colframe=primary, arc=3mm, boxrule=1pt, center]",
        r"\centering\sffamily\small",
        r"\textbf{涵盖 14 篇系统理论全景推导 + 14 组完整可运行实验源码精解}\\[0.3cm]",
        r"Transformer $\cdot$ RoPE $\cdot$ SwiGLU $\cdot$ GQA/MLA $\cdot$ PagedAttention $\cdot$ Speculative Decoding $\cdot$ Tensor Parallelism",
        r"\end{tcolorbox}",
        r"",
        r"\vfill",
        r"{\large\sffamily sharp-volta 开源系统学习工程组 编著}\\[0.3cm]",
        r"{\small\sffamily 2026 年 9 月 $\cdot$ 第一版}",
        r"\vspace*{1.5cm}",
        r"\end{titlepage}",
        r"",
        r"% ----------------------------------------------------------------------",
        r"% 前言与目录",
        r"% ----------------------------------------------------------------------",
        r"\frontmatter",
        r"\pagestyle{plain}",
        r"\include{chapters/preface}",
        r"\tableofcontents",
        r"\cleardoublepage",
        r"",
        r"% ----------------------------------------------------------------------",
        r"% 正文主体",
        r"% ----------------------------------------------------------------------",
        r"\mainmatter",
        r"\pagestyle{fancy}",
        r"",
        r"\part{核心原理与数学推导篇 (Theory \& Derivations)}",
    ]
    
    for f in part1_files:
        cname = os.path.splitext(f)[0]
        main_tex.append(f"\\include{{chapters/{cname}}}")
        
    main_tex.append("")
    main_tex.append(r"\part{工业实战与实验源码精解篇 (Practical Labs \& Implementations)}")
    
    for f in part2_files:
        lname = os.path.splitext(f)[0]
        main_tex.append(f"\\include{{labs/{lname}}}")
        
    main_tex.append("")
    main_tex.append(r"\end{document}")
    
    main_path = os.path.join(OUTPUT_DIR, "main.tex")
    with open(main_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(main_tex))
        
    print(f"✅ 生成完成！主控文件位于: {main_path}")
    print(f"   第一部分章节: {len(part1_files)} 篇")
    print(f"   第二部分实战: {len(part2_files)} 组")

if __name__ == "__main__":
    main()
