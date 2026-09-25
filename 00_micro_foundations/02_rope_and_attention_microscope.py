"""
模块 00 - 脚本 2: 旋转位置编码 (RoPE) 与自注意力因果掩码极细拆解

深入解答两个核心疑问：
1. 【为什么必须有位置编码？】：自注意力公式 QK^T 是置换不变的（词袋）。打乱输入顺序，计算出的词嵌入数值完全不改变！
2. 【RoPE 怎么旋转向量的？】：为什么现代大模型（LLaMA/Qwen/DeepSeek）全都在用 RoPE？
   - 它是如何把向量拆成 2 维对，在复平面旋转角度 m*theta 的？
   - 为什么旋转后点积只取决于相对距离 (m - n)？
"""

import math


def print_title(title):
    print("\n" + "=" * 74)
    print(f"🔬 {title}")
    print("=" * 74)


# =====================================================================
# 1. 揭秘：没有位置编码时，Attention 为什么是“瞎子”？
# =====================================================================
def demo_permutation_invariance():
    print_title("1. 为什么自注意力机制本身'没有语序感'？")
    
    # 假设有三个词的 Query 向量
    # 句子 A: "猫 吃 鱼"
    # 句子 B: "鱼 吃 猫"
    q_cat  = [1.0, 0.0]
    q_eat  = [0.0, 1.0]
    q_fish = [0.8, 0.6]

    print("如果只做简单的点积 Q · K^T：")
    dot_cat_fish = q_cat[0]*q_fish[0] + q_cat[1]*q_fish[1]
    dot_fish_cat = q_fish[0]*q_cat[0] + q_fish[1]*q_cat[1]

    print(f"   - '猫' 与 '鱼' 的点积: {dot_cat_fish:.4f}")
    print(f"   - '鱼' 与 '猫' 的点积: {dot_fish_cat:.4f}")
    print("👉 结论: 在没有任何位置编码的情况下，'猫吃鱼' 和 '鱼吃猫' 在数学上完全无法区分！")
    print("   因此必须显式地给每个 Token 注入其在序列中的位置坐标 m (m=0, 1, 2, ...)。")


# =====================================================================
# 2. RoPE (Rotary Position Embedding) 旋转实操
# =====================================================================
def apply_rope_2d(x, m, base_theta=10000.0):
    """
    针对一个 2 维向量 x = [x0, x1]，在位置 m 处逆时针旋转角度 theta_m
    theta_m = m / (base_theta ^ (0 / 2)) = m * 1.0
    旋转矩阵:
    [ cos(theta), -sin(theta) ]
    [ sin(theta),  cos(theta) ]
    """
    theta = m * (1.0 / (base_theta ** (0.0 / 2.0)))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    x0_rot = x[0] * cos_t - x[1] * sin_t
    x1_rot = x[0] * sin_t + x[1] * cos_t
    return [x0_rot, x1_rot], theta


def demo_rope_mathematical_magic():
    print_title("2. RoPE 旋转位置编码的微观数学推导与相对位置特性")
    
    # 一个基准向量
    v = [1.0, 0.0] # 长度为 1 的水平向量
    
    print(f"原始向量 v = {v} (模长 = 1.0)")
    print("\n在不同位置 m 处的旋转效果:")
    
    rotated_vectors = []
    positions = [0, 1, 2, 3]
    for pos in positions:
        v_rot, theta = apply_rope_2d(v, pos, base_theta=10.0)
        norm = math.sqrt(v_rot[0]**2 + v_rot[1]**2)
        print(f"   - 位置 m={pos}: 旋转角度 = {math.degrees(theta):5.1f}° | 旋转后向量 = [{v_rot[0]:6.3f}, {v_rot[1]:6.3f}] | 模长保持: {norm:.4f}")
        rotated_vectors.append(v_rot)

    print("\n✨ 核心神奇特性：点积仅取决于相对距离 (Relative Distance)！")
    # 计算 (位置 1 和 位置 2) 的点积 (相对距离 = 1)
    dot_1_2 = rotated_vectors[1][0]*rotated_vectors[2][0] + rotated_vectors[1][1]*rotated_vectors[2][1]
    
    # 计算 (位置 2 和 位置 3) 的点积 (相对距离 = 1)
    dot_2_3 = rotated_vectors[2][0]*rotated_vectors[3][0] + rotated_vectors[2][1]*rotated_vectors[3][1]
    
    # 计算 (位置 0 和 位置 1) 的点积 (相对距离 = 1)
    dot_0_1 = rotated_vectors[0][0]*rotated_vectors[1][0] + rotated_vectors[0][1]*rotated_vectors[1][1]
    
    # 计算 (位置 0 和 位置 3) 的点积 (相对距离 = 3)
    dot_0_3 = rotated_vectors[0][0]*rotated_vectors[3][0] + rotated_vectors[0][1]*rotated_vectors[3][1]

    print(f"   - 距离 Δ=1: 点积(位置0, 位置1) = {dot_0_1:.6f}")
    print(f"   - 距离 Δ=1: 点积(位置1, 位置2) = {dot_1_2:.6f}")
    print(f"   - 距离 Δ=1: 点积(位置2, 位置3) = {dot_2_3:.6f}")
    print(f"   - 距离 Δ=3: 点积(位置0, 位置3) = {dot_0_3:.6f}")

    print("\n💡 惊人发现:")
    print("   1. 只要两个词之间的【相对距离】都是 1，无论它们位于文章的第 0、第 100 还是第 1000 个位置，")
    print(f"      它们经 RoPE 旋转后的注意力点积完全相同 ({dot_0_1:.6f})！")
    print("   2. 旋转操作（正交变换）完全不改变向量本身的模长（范数不变性）。")
    print("   3. 随着距离变远（如 Δ=3 时点积变为 {dot_0_3:.4f}），注意力自然发生长程衰减，符合人类阅读理解习惯。")


# =====================================================================
# 3. 因果掩码 (Causal Mask) 数值实验
# =====================================================================
def demo_causal_mask_simulation():
    print_title("3. 因果掩码 (Causal Mask) 到底怎么做到'禁止向后看'？")
    
    raw_scores = [
        [2.0, 1.5, 3.0], # 词 0 对词 0, 1, 2 的原始点积
        [1.0, 2.5, 1.8], # 词 1 对词 0, 1, 2 的原始点积
        [0.5, 1.2, 2.2], # 词 2 对词 0, 1, 2 的原始点积
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
                row.append(-1e9) # -infinity
            else:
                row.append(raw_scores[i][j])
        masked_scores.append(row)
        
    for row in masked_scores:
        row_display = [f"{x:6.1f}" if x > -1e8 else "  -inf" for x in row]
        print("   [" + ", ".join(row_display) + "]")

    print("\n经过 Softmax 归一化后的真实注意力权重矩阵 (每一行和为 1.0):")
    for r in range(3):
        max_v = max(masked_scores[r])
        exp_v = [math.exp(x - max_v) for x in masked_scores[r]]
        sum_e = sum(exp_v)
        probs = [e / sum_e for e in exp_v]
        print(f"   第 {r} 个词的注意力分布 -> " + str([f"{p*100:5.1f}%" for p in probs]))


if __name__ == "__main__":
    demo_permutation_invariance()
    demo_rope_mathematical_magic()
    demo_causal_mask_simulation()
