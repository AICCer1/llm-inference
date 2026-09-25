"""
模块 03: 投机采样 (Speculative Decoding) 核心逻辑算法仿真

实现无偏差投机采样 (Lossless Speculative Sampling):
1. 草稿模型 (Draft Model) 连续自回归推测 K 个候选 Token
2. 目标大模型 (Target Model) 并行前向传播一次，获取各位置的条件分布
3. 执行严格的概率接收/拒绝准则 (Leviathan et al.)
4. 统计接受率 (Acceptance Rate)、单步有效产出 (Tokens per Step) 与加速比
"""

import random
import time
import math


class ToyLanguageModel:
    """模拟语言模型，输出词表上的概率分布"""
    def __init__(self, vocab_size=1000, forward_latency_ms=10.0, seed=42):
        self.vocab_size = vocab_size
        self.latency_s = forward_latency_ms / 1000.0
        self.rng = random.Random(seed)

    def sample_distribution(self, context_tokens):
        """模拟针对当前上下文输出的下一个 Token 概率分布 (Dirichlet/Softmax)"""
        # 利用上下文 hash 保证相同前缀产生相同的确定性先验，加上随机噪声
        ctx_hash = sum(context_tokens) % 10007
        raw_logits = [(math.sin(i * 0.1 + ctx_hash) + 1.0) for i in range(self.vocab_size)]
        
        # 针对前几个词加大权重，模拟自然语言的长尾分布
        for i in range(min(10, self.vocab_size)):
            raw_logits[i] *= 2.5

        sum_l = sum(raw_logits)
        return [l / sum_l for l in raw_logits]

    def sample_next_token(self, prob_dist):
        """根据分布依概率采样一个 Token"""
        r = random.random()
        cumulative = 0.0
        for idx, p in enumerate(prob_dist):
            cumulative += p
            if r <= cumulative:
                return idx
        return len(prob_dist) - 1


def speculative_decoding_simulation(
    target_model_latency=30.0, # 大模型单次前向 30ms (如 70B)
    draft_model_latency=3.0,   # 小模型单次前向 3ms (如 7B 或专门的草稿头)
    gamma=4,                   # 每轮草稿模型推测 token 数 K=4
    total_tokens_to_generate=50,
    similarity=0.75            # 草稿模型与目标模型分布的相似度/拟合程度
):
    print("=" * 68)
    print("🚀 投机采样 (Speculative Decoding) 端到端仿真")
    print(f"📌 参数配置: 大模型耗时={target_model_latency}ms | 草稿模型耗时={draft_model_latency}ms")
    print(f"📌 投机窗口大小 K={gamma} | 目标生成长度={total_tokens_to_generate} | 模型相似度={similarity*100:.0f}%")
    print("=" * 68 + "\n")

    vocab_size = 200
    target_lm = ToyLanguageModel(vocab_size=vocab_size, forward_latency_ms=target_model_latency, seed=1)
    
    generated_tokens = [1, 2, 3] # 初始 Prompt
    
    round_count = 0
    total_accepted_tokens = 0
    simulated_wall_time = 0.0
    tokens_produced = 0

    print("开始投机解码迭代...\n")

    while tokens_produced < total_tokens_to_generate:
        round_count += 1
        
        # -------------------------------------------------------------
        # 阶段 1: 草稿模型连续自回归预测 gamma 个 token
        # -------------------------------------------------------------
        draft_tokens = []
        draft_probs = []
        curr_draft_context = list(generated_tokens)
        
        for _ in range(gamma):
            # 草稿模型前向
            simulated_wall_time += draft_model_latency
            
            # 模拟草稿分布（以大模型为基准，加入适度偏置模拟小模型的精度损失）
            true_dist = target_lm.sample_distribution(curr_draft_context)
            draft_dist = []
            for p in true_dist:
                # 混合真实分布与均匀分布
                p_draft = similarity * p + (1.0 - similarity) * (1.0 / vocab_size)
                draft_dist.append(p_draft)
            
            s = sum(draft_dist)
            draft_dist = [p / s for p in draft_dist]
            
            next_t = target_lm.sample_next_token(draft_dist)
            draft_tokens.append(next_t)
            draft_probs.append(draft_dist[next_t])
            curr_draft_context.append(next_t)

        # -------------------------------------------------------------
        # 阶段 2: 目标大模型单次前向，并行验证所有 gamma 个 token
        # -------------------------------------------------------------
        # 注意：这里只花费 1 次大模型前向的时间！
        simulated_wall_time += target_model_latency
        
        # 逐个位置进行接收/拒绝检验 (Leviathan et al. 准则)
        accepted_this_round = []
        curr_verify_context = list(generated_tokens)
        
        all_accepted = True
        bonus_token = None
        
        for i in range(gamma):
            cand_token = draft_tokens[i]
            p_draft = draft_probs[i]
            
            # 大模型计算该上下文下的真实条件概率
            target_dist = target_lm.sample_distribution(curr_verify_context)
            p_target = target_dist[cand_token]
            
            # 接收概率 alpha = min(1, p_target / p_draft)
            alpha = min(1.0, p_target / p_draft if p_draft > 0 else 1.0)
            
            if random.random() < alpha:
                # 接受该候选词
                accepted_this_round.append(cand_token)
                curr_verify_context.append(cand_token)
            else:
                # 拒绝该候选词，并根据残差分布从大模型无偏重采样一个替代词
                all_accepted = False
                # 残差分布: max(0, p_target - p_draft)
                residual = [max(0.0, target_dist[v] - (similarity * target_dist[v] + (1 - similarity)/vocab_size)) for v in range(vocab_size)]
                sum_res = sum(residual)
                if sum_res > 1e-9:
                    norm_residual = [r / sum_res for r in residual]
                    replacement_token = target_lm.sample_next_token(norm_residual)
                else:
                    replacement_token = target_lm.sample_next_token(target_dist)
                
                accepted_this_round.append(replacement_token)
                break
                
        # 如果全部 gamma 个词都被接受，大模型可以“免费”多采样 1 个新词！
        if all_accepted:
            final_dist = target_lm.sample_distribution(curr_verify_context)
            bonus_token = target_lm.sample_next_token(final_dist)
            accepted_this_round.append(bonus_token)

        # 更新已生成序列
        generated_tokens.extend(accepted_this_round)
        tokens_produced += len(accepted_this_round)
        total_accepted_tokens += len(accepted_this_round)

        print(f"Round {round_count:2d}: 草稿推测 {gamma} 词 -> 成功采纳 {len(accepted_this_round)} 词 (Tokens: {accepted_this_round})")

    # -------------------------------------------------------------
    # 阶段 3: 对比基线（无投机采样，单独用大模型逐字生成）
    # -------------------------------------------------------------
    baseline_wall_time = tokens_produced * target_model_latency
    avg_tokens_per_round = total_accepted_tokens / round_count
    speedup = baseline_wall_time / simulated_wall_time

    print("\n" + "=" * 68)
    print("📊 仿真测试统计结果")
    print("=" * 68)
    print(f"   - 总生成 Token 数量        : {tokens_produced}")
    print(f"   - 大模型总前向次数 (Rounds) : {round_count} 次 (若无投机需要 {tokens_produced} 次)")
    print(f"   - 平均每轮产生 Token 数    : {avg_tokens_per_round:.2f} tokens/step")
    print(f"   - 基线耗时 (仅大模型生成)   : {baseline_wall_time:.1f} ms")
    print(f"   - 投机采样累计耗时         : {simulated_wall_time:.1f} ms")
    print(f"   - 🚀 端到端真实加速比       : {speedup:.2f}x")
    print("=" * 68)
    print("💡 结论:")
    print("   投机采样的加速比核心取决于：")
    print("   1. 草稿模型的接受率 (Acceptance Rate)；")
    print("   2. 草稿模型与大模型的相对耗时比 (越轻量越好)。")
    print("   在现代框架（如 vLLM / SGLang）中，投机采样最高可达 2.0x ~ 3.0x 的吞吐提升！")


if __name__ == "__main__":
    speculative_decoding_simulation()
