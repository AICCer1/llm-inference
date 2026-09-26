"""
Lab 04 - 脚本 2: 旋转位置编码 (RoPE) 与自注意力因果掩码极细拆解

深入解答三个核心疑问：
1. 【为什么必须有位置编码？】：没有位置编码的自注意力是"置换等变"的——
   把输入顺序打乱，每个词得到的输出向量完全不变，只是跟着换了个位置。
   所以 "猫 吃 鱼" 和 "鱼 吃 猫" 里的 "猫"，模型看到的是同一个东西。
2. 【RoPE 怎么旋转向量的？】：把向量拆成 2 维对，在复平面按位置 m 旋转 m*theta_i，
   旋转后点积只取决于相对距离 (m - n)。
3. 【长程衰减从哪来？】：单个频率下点积 = cos(Δθ)，是周期函数，根本不衰减；
   只有把很多个不同频率 theta_i 叠加起来，远距离处各分量相位错开、互相抵消，才出现"平均意义上"的衰减。
"""

import math
import random


def print_title(title):
    print("\n" + "=" * 74)
    print(f"🔬 {title}")
    print("=" * 74)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(row):
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]


def self_attention_no_pos(X, Wq, Wk, Wv):
    """最朴素的单头自注意力（无位置编码、无掩码），X: [T, d]"""
    def proj(x, W):
        return [sum(x[i] * W[i][j] for i in range(len(x))) for j in range(len(W[0]))]
    Q = [proj(x, Wq) for x in X]
    K = [proj(x, Wk) for x in X]
    V = [proj(x, Wv) for x in X]
    d = len(Q[0])
    out = []
    for q in Q:
        w = softmax([dot(q, k) / math.sqrt(d) for k in K])
        out.append([sum(w[j] * V[j][c] for j in range(len(V))) for c in range(d)])
    return out


# =====================================================================
# 1. 揭秘：没有位置编码时，Attention 为什么是"瞎子"？
# =====================================================================
def demo_permutation_equivariance():
    print_title("1. 为什么自注意力机制本身'没有语序感'？(置换等变性)")

    rng = random.Random(0)
    d = 4
    emb = {w: [rng.gauss(0, 1) for _ in range(d)] for w in ["猫", "吃", "鱼"]}
    Wq = [[rng.gauss(0, 0.5) for _ in range(d)] for _ in range(d)]
    Wk = [[rng.gauss(0, 0.5) for _ in range(d)] for _ in range(d)]
    Wv = [[rng.gauss(0, 0.5) for _ in range(d)] for _ in range(d)]

    s1 = ["猫", "吃", "鱼"]
    s2 = ["鱼", "吃", "猫"]
    out1 = self_attention_no_pos([emb[w] for w in s1], Wq, Wk, Wv)
    out2 = self_attention_no_pos([emb[w] for w in s2], Wq, Wk, Wv)

    print("分别把 '猫 吃 鱼' 与 '鱼 吃 猫' 送入同一个【无位置编码】的自注意力层：")
    for w in ["猫", "鱼"]:
        v1 = out1[s1.index(w)]
        v2 = out2[s2.index(w)]
        diff = max(abs(a - b) for a, b in zip(v1, v2))
        print(f"   '{w}' 在句子1中的输出: {[round(v, 4) for v in v1]}")
        print(f"   '{w}' 在句子2中的输出: {[round(v, 4) for v in v2]}   (最大差值 {diff:.1e})")
        assert diff == 0.0, '无位置编码时同一个词的输出应与位置无关'

    print("👉 结论: 同一个词无论在句首还是句尾，得到的输出向量【完全一样】，只是换了个位置。")
    print("   原因: 注意力对 K/V 做的是加权求和 Σ_j w_j v_j，求和与 j 的排列顺序无关。")
    print("   所以必须显式地给每个 Token 注入位置 m，否则 '猫吃鱼' 与 '鱼吃猫' 在模型眼里是同一个词袋。")


# =====================================================================
# 2. RoPE (Rotary Position Embedding) 旋转实操
# =====================================================================
def rope(x, m, base=10000.0):
    """
    对 d 维向量 x（d 为偶数）在位置 m 处施加 RoPE：
    第 i 对分量 (x[2i], x[2i+1]) 旋转角度 m * theta_i，theta_i = base^(-2i/d)
    （与 Llama 的实现在"如何配对"上略有不同：HF 实现把前一半与后一半配对，数学性质完全相同）
    """
    d = len(x)
    out = [0.0] * d
    for i in range(d // 2):
        theta = base ** (-2.0 * i / d)
        c, s = math.cos(m * theta), math.sin(m * theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


def demo_rope_relative_property():
    print_title("2. RoPE: 旋转后点积只取决于相对距离 (m - n)")

    rng = random.Random(1)
    d = 8
    q = [rng.gauss(0, 1) for _ in range(d)]
    k = [rng.gauss(0, 1) for _ in range(d)]

    print(f"随机取一对 {d} 维向量 q、k；模长 |q| = {math.sqrt(dot(q, q)):.4f}")
    for m in [0, 5, 100]:
        print(f"   RoPE(q, m={m:<3d}) 的模长 = {math.sqrt(dot(rope(q, m), rope(q, m))):.4f}   (旋转是正交变换，不改变模长)")

    print("\n固定相对距离 Δ = m - n = 3，改变绝对位置：")
    for n in [0, 10, 1000, 50000]:
        m = n + 3
        val = dot(rope(q, m), rope(k, n))
        print(f"   <RoPE(q, {m:>5d}), RoPE(k, {n:>5d})> = {val:+.6f}")
        assert abs(val - dot(rope(q, 3), k)) < 1e-9, "RoPE 点积应只依赖相对距离"
    print("👉 绝对位置从 0 变到 50000，点积一字不差 —— 只剩下相对距离 Δ 在起作用。")


def demo_rope_long_range_decay():
    print_title("3. 长程衰减：只在 q、k 相关时的【期望】上成立")

    print("(a) 只有 1 对分量（单一频率 θ=1）时，q=k=[1,0]：点积 = cos(Δ)，是周期函数：")
    for delta in [0, 1, 2, 3, 6, 7]:
        val = dot(rope([1.0, 0.0], delta, base=1.0), [1.0, 0.0])
        print(f"   Δ={delta:<3d} 点积 = {val:+.4f}")
    print("   👉 Δ=3 时是 -0.99，Δ=6 又回到 +0.96 —— 单一频率【没有】长程衰减。")

    d, n_pairs = 64, 300
    rng = random.Random(2)
    deltas = [0, 4, 16, 64, 256, 1024, 4096]
    qs = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n_pairs)]
    noise = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n_pairs)]
    cases = {
        "k = q（完全相关）": [q[:] for q in qs],
        "k = 0.5q + 噪声": [[0.5 * a + 0.866 * b for a, b in zip(q, z)] for q, z in zip(qs, noise)],
        "q、k 独立随机": noise,
    }
    print(f"\n(b) d={d}（32 个频率），每种情况 {n_pairs} 对随机向量，统计 RoPE 点积随距离 Δ 的变化：")
    print("   " + f"{'':<16}" + "".join(f"{'Δ=' + str(D):>9}" for D in deltas))
    for name, ks in cases.items():
        vals = {D: [dot(rope(q, D), k) for q, k in zip(qs, ks)] for D in deltas}
        mean = [sum(v) / n_pairs for v in vals.values()]
        std = [math.sqrt(sum((x - m) ** 2 for x in v) / n_pairs) for v, m in zip(vals.values(), mean)]
        print(f"   {name:<14} 均值" + "".join(f"{m:9.2f}" for m in mean))
        se = max(std) / math.sqrt(n_pairs)   # 均值的标准误
        if name.startswith("q、k 独立"):
            assert all(abs(m) < 5 * se for m in mean), "独立随机时各距离的均值应 ≈ 0（无衰减）"
        else:
            assert mean[0] - mean[-2] > 10 * se, "相关时均值应随距离明显衰减"
        print(f"   {'':<14} 标准差" + "".join(f"{s_:9.2f}" for s_ in std))
    print("   👉 q、k 相关时，点积的【期望】随距离衰减：同样'匹配'的两个词，离得越远注意力加成越小。")
    print("      q、k 独立时，均值始终 ≈ 0、标准差始终 ≈ √d：旋转不改变各向同性随机向量的分布，【没有任何衰减】。")
    print("      RoFormer 论文 §3.4.3 证明的是一个【上界】的衰减，不是点积本身一定变小。")
    print("      所以 RoPE 的衰减是对'匹配信号'的软性偏置，不是硬性的距离惩罚。")


# =====================================================================
# 4. 因果掩码 (Causal Mask) 数值实验
# =====================================================================
def demo_causal_mask_simulation():
    print_title("4. 因果掩码 (Causal Mask) 到底怎么做到'禁止向后看'？")

    raw_scores = [
        [2.0, 1.5, 3.0],  # 词 0 对词 0, 1, 2 的原始点积
        [1.0, 2.5, 1.8],  # 词 1 对词 0, 1, 2 的原始点积
        [0.5, 1.2, 2.2],  # 词 2 对词 0, 1, 2 的原始点积
    ]

    print("原始未经掩码的点积得分矩阵 (未遮挡未来):")
    for row in raw_scores:
        print("   " + str([round(x, 2) for x in row]))

    print("\n叠加因果下三角掩码 (-inf 遮挡未来):")
    masked_scores = []
    for i in range(3):
        row = []
        for j in range(3):
            if j > i:
                row.append(-1e9)  # -infinity
            else:
                row.append(raw_scores[i][j])
        masked_scores.append(row)

    for row in masked_scores:
        row_display = [f"{x:6.1f}" if x > -1e8 else "  -inf" for x in row]
        print("   [" + ", ".join(row_display) + "]")

    print("\n经过 Softmax 归一化后的真实注意力权重矩阵 (每一行和为 1.0):")
    for r in range(3):
        probs = softmax(masked_scores[r])
        print(f"   第 {r} 个词的注意力分布 -> " + str([f"{p*100:5.1f}%" for p in probs]))


if __name__ == "__main__":
    demo_permutation_equivariance()
    demo_rope_relative_property()
    demo_rope_long_range_decay()
    demo_causal_mask_simulation()
