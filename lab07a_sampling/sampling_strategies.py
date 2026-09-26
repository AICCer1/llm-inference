"""
Lab 07a: 采样策略 (Sampling Strategies) 微观实验

深入拆解模型生成下一个 Token 时的四大核心采样算法：
1. 贪婪搜索 (Greedy Search)
2. 温度系数调节 (Temperature Scaling)
3. Top-K 截断采样
4. Top-P (核采样 Nucleus Sampling)
5. 惩罚项机制 (Repetition Penalty 重复惩罚)
"""

import math
import random


def softmax(logits):
    max_l = max(logits)
    exp_l = [math.exp(x - max_l) for x in logits]
    sum_e = sum(exp_l)
    return [e / sum_e for e in exp_l]


def print_distribution_table(vocab_words, probs, title):
    print(f"\n📊 {title}")
    # 按概率从大到小排序
    ranked = sorted(zip(vocab_words, probs), key=lambda x: x[1], reverse=True)
    for word, p in ranked:
        bar = "█" * int(p * 40)
        print(f"   {word:<6} | {p*100:5.2f}% | {bar}")


# =====================================================================
# 1. 温度系数 (Temperature)
# =====================================================================
def apply_temperature(logits, temperature=1.0):
    """
    Temperature 调整: logits_new = logits / T
    - T < 1.0 (如 0.2): 分布尖锐化，增强确定性、严肃性
    - T = 1.0: 原始自然分布
    - T > 1.0 (如 1.8): 分布均匀化，增强创造力、不可预测性
    """
    if temperature <= 1e-4:
        # 退化为贪婪模式
        max_idx = logits.index(max(logits))
        return [1.0 if i == max_idx else 0.0 for i in range(len(logits))]
    
    scaled_logits = [l / temperature for l in logits]
    return softmax(scaled_logits)


# =====================================================================
# 2. Top-K 采样
# =====================================================================
def apply_top_k(logits, k=3):
    """
    仅保留概率/打分最高的前 K 个 Token，其余位置全部置为 -inf
    """
    # 按分数排序后取前 K 个【下标】。不能用"分数 >= 第 K 大的值"做阈值：
    # 有并列时会多选，例如 [1, 1, 1, 0] 取 K=2 会留下 3 个。并列时按下标先后取（与 torch.topk 的常见行为一致）。
    keep = set(sorted(range(len(logits)), key=lambda i: -logits[i])[:k])
    filtered_logits = [l if i in keep else -1e9 for i, l in enumerate(logits)]
    return softmax(filtered_logits)


# =====================================================================
# 3. Top-P (核采样 Nucleus Sampling)
# =====================================================================
def apply_top_p(logits, p=0.85):
    """
    将候选词按概率从大到小排序，累加其概率值。
    一旦累计概率超过阈值 P，立即截断后续的长尾无意义词，并在保留集合中重新归一化。
    """
    raw_probs = softmax(logits)
    # 带索引排序
    indexed_probs = sorted(enumerate(raw_probs), key=lambda x: x[1], reverse=True)
    
    cum_sum = 0.0
    keep_indices = set()
    for idx, prob in indexed_probs:
        keep_indices.add(idx)
        cum_sum += prob
        if cum_sum >= p:
            break
            
    # 构建截断后的 logits
    filtered_logits = []
    for i, l in enumerate(logits):
        if i in keep_indices:
            filtered_logits.append(l)
        else:
            filtered_logits.append(-1e9)
            
    return softmax(filtered_logits)


# =====================================================================
# 4. 重复惩罚 (Repetition Penalty)
# =====================================================================
def apply_repetition_penalty(logits, past_token_indices, penalty=1.2):
    """
    对已经在前文中出现过的 Token 施加打分惩罚，降低连续复读概率
    - 若 logit > 0: logit = logit / penalty
    - 若 logit < 0: logit = logit * penalty
    """
    new_logits = list(logits)
    for idx in past_token_indices:
        l = new_logits[idx]
        if l > 0:
            new_logits[idx] = l / penalty
        else:
            new_logits[idx] = l * penalty
    return new_logits


def run_experiment():
    print("=" * 72)
    print("🔬 大模型采样算法 (Sampling Strategies) 微观数值对比")
    print("=" * 72)

    # 假定模型输出的一组候选词未归一化 Logits
    words  = ["北京", "上海", "广州", "深圳", "月球", "香蕉"]
    # 模型认为"北京"最可能，但"月球"和"香蕉"也有微小的可能噪音
    logits = [ 3.2,    2.8,    1.5,    1.2,   -1.0,   -2.5 ]

    # 0. 贪婪搜索：直接取 argmax，不涉及概率
    greedy = max(range(len(logits)), key=lambda i: logits[i])
    print(f"\n📊 0. 贪婪搜索 (Greedy)：argmax(logits) = '{words[greedy]}'，每次都选它，结果完全确定")

    # 1. 原始 Softmax 分布
    p_orig = softmax(logits)
    print_distribution_table(words, p_orig, "1. 原始分布 (Temperature = 1.0, 默认自然状态)")

    # 2. 低温 (T = 0.2)
    p_cold = apply_temperature(logits, temperature=0.2)
    print_distribution_table(words, p_cold, "2. 低温严肃模式 (Temperature = 0.2 -> 头部概率暴涨)")

    # 3. 高温 (T = 2.0)
    p_hot = apply_temperature(logits, temperature=2.0)
    print_distribution_table(words, p_hot, "3. 高温创造模式 (Temperature = 2.0 -> 分布平坦，冷门词被激发)")

    # 4. Top-K (K=2)
    p_topk = apply_top_k(logits, k=2)
    print_distribution_table(words, p_topk, "4. Top-K 截断 (K = 2 -> 仅保留最高前两名，其余全部剔除)")

    # 5. Top-P (P=0.85)
    p_topp = apply_top_p(logits, p=0.85)
    print_distribution_table(words, p_topp, "5. Top-P 核采样 (P = 0.85 -> 动态累加截断长尾有害词)")

    # 6. 重复惩罚测试 (假设模型刚说完"北京"，惩罚已出现的词)
    print("\n" + "=" * 72)
    print("6. 重复惩罚机制 (Repetition Penalty = 1.5):")
    penalized_logits = apply_repetition_penalty(logits, past_token_indices=[0], penalty=1.5)
    p_penalized = softmax(penalized_logits)
    print(f"   对历史词 '北京' 实施 1.5 倍惩罚后:")
    print_distribution_table(words, p_penalized, "施加惩罚后的分布 (北京的概率被主动压低，避免复读机现象)")
    print("=" * 72)


def self_check():
    """自检：分布合法、Top-K 恰好保留 K 个（含并列情形）、Top-P 保留的是概率最高的若干个"""
    for probs in [apply_temperature([3.2, 2.8, 1.5], t) for t in (0.2, 1.0, 2.0)] + [apply_top_p([3.2, 2.8, 1.5, -1.0], 0.85)]:
        assert abs(sum(probs) - 1) < 1e-9 and min(probs) >= 0
    tie = apply_top_k([1.0, 1.0, 1.0, 0.0], k=2)
    assert sum(p > 1e-6 for p in tie) == 2, "并列时 Top-K 不应多选"
    assert sum(p > 1e-6 for p in apply_top_k([3.2, 2.8, 1.5, 1.2, -1.0], k=3)) == 3


if __name__ == "__main__":
    self_check()
    run_experiment()
