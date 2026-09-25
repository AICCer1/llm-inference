"""
模块 00 - 脚本 1: 大模型推理前向传播「显微镜级」单步跟踪器 (Zero-Dependency Pure Python)

本脚本为你彻底揭开大模型推理的黑盒：
输入一段极小文本: ["大", "模", "型"] (Token IDs: [5, 6, 7])
逐行打印每个算子的张量形状（Shape）、物理含义、因果注意力权重图与 Logits 排行榜。
包含 Prefill 首字阶段与 Decode 增量阶段的真实计算过程。
"""

import math
import random


# =====================================================================
# 1. 微型分词器与词表 (Micro Tokenizer & Vocab)
# =====================================================================
VOCAB = {
    0: "<pad>",
    1: "我",
    2: "爱",
    3: "学",
    4: "习",
    5: "大",
    6: "模",
    7: "型",
    8: "推",
    9: "理",
    10: "<eos>"
}
WORD2ID = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

# 模型超参数 (超微型配置，方便逐个数字打印查看)
D_MODEL = 16       # 隐藏层维度
N_HEADS = 2        # 注意力头数
D_HEAD = D_MODEL // N_HEADS # 每个头的维度: 16 // 2 = 8
HIDDEN_DIM = 32    # FFN 中间层维度


# =====================================================================
# 2. 基础张量算子实现 (手写纯 Python 矩阵运算，无任何黑盒)
# =====================================================================
def create_random_matrix(rows, cols, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 0.2) for _ in range(cols)] for _ in range(rows)]

def matmul_2d(A, B):
    """[N, K] x [K, M] -> [N, M]"""
    n, k = len(A), len(A[0])
    m = len(B[0])
    res = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for p in range(k):
            val = A[i][p]
            for j in range(m):
                res[i][j] += val * B[p][j]
    return res

def rms_norm(X, weight, eps=1e-5):
    """
    RMSNorm: 均方根归一化
    X: [seq_len, d_model]
    weight: [d_model]
    """
    out = []
    for row in X:
        mean_square = sum(x * x for x in row) / len(row)
        scale = 1.0 / math.sqrt(mean_square + eps)
        normed_row = [x * scale * w for x, w in zip(row, weight)]
        out.append(normed_row)
    return out

def silu(x):
    """SiLU (Swish) 激活函数: x * sigmoid(x)"""
    return x / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

def softmax(row):
    """防溢出 Softmax"""
    max_val = max(row)
    exp_vals = [math.exp(v - max_val) for v in row]
    sum_exp = sum(exp_vals)
    return [v / sum_exp for v in exp_vals]


# =====================================================================
# 3. 显微镜跟踪类
# =====================================================================
class MicroLlamaTracer:
    def __init__(self):
        # 权重初始化
        self.embed_table = create_random_matrix(VOCAB_SIZE, D_MODEL, seed=1) # [11, 16]
        self.norm1_w = [1.0] * D_MODEL
        self.w_q = create_random_matrix(D_MODEL, D_MODEL, seed=2) # [16, 16]
        self.w_k = create_random_matrix(D_MODEL, D_MODEL, seed=3)
        self.w_v = create_random_matrix(D_MODEL, D_MODEL, seed=4)
        self.w_o = create_random_matrix(D_MODEL, D_MODEL, seed=5)
        
        self.norm2_w = [1.0] * D_MODEL
        # SwiGLU 权重
        self.w_gate = create_random_matrix(D_MODEL, HIDDEN_DIM, seed=6)
        self.w_up   = create_random_matrix(D_MODEL, HIDDEN_DIM, seed=7)
        self.w_down = create_random_matrix(HIDDEN_DIM, D_MODEL, seed=8)
        
        self.final_norm_w = [1.0] * D_MODEL
        self.lm_head = create_random_matrix(D_MODEL, VOCAB_SIZE, seed=9) # [16, 11]

    def forward_trace(self, token_ids, phase_name="Prefill"):
        T = len(token_ids)
        tokens_str = [VOCAB[tid] for tid in token_ids]
        
        print("\n" + "=" * 76)
        print(f"🔬 【{phase_name} 阶段】显微镜单步追踪")
        print(f"📥 输入文本: {' + '.join(tokens_str)}  (Token IDs: {token_ids})")
        print(f"📐 序列长度 T = {T} | 隐藏层维度 D = {D_MODEL} | 头数 Heads = {N_HEADS} (单头维度 {D_HEAD})")
        print("=" * 76)

        # -------------------------------------------------------------
        # Step 1: Embedding 查表
        # -------------------------------------------------------------
        x = [self.embed_table[tid][:] for tid in token_ids] # [T, D_MODEL]
        print(f"\n[Step 1] Embedding 词嵌入查表:")
        print(f"         输入形状: [{T}]  -->  输出形状: [{T}, {D_MODEL}]")
        print(f"         说明: 每一个整数 Token ID 被映射为 {D_MODEL} 维稠密特征向量。")
        print(f"         示例 (Token '{tokens_str[0]}' 前 4 维特征): {[round(v, 3) for v in x[0][:4]]}...")

        # -------------------------------------------------------------
        # Step 2: 第 1 层 RMSNorm
        # -------------------------------------------------------------
        x_norm1 = rms_norm(x, self.norm1_w)
        print(f"\n[Step 2] Attention 前 RMSNorm 归一化:")
        print(f"         输入形状: [{T}, {D_MODEL}]  -->  输出形状: [{T}, {D_MODEL}]")
        print(f"         说明: 消除特征幅值异常波动，稳定注意力输入方差。")

        # -------------------------------------------------------------
        # Step 3: Q, K, V 投影矩阵乘法
        # -------------------------------------------------------------
        Q = matmul_2d(x_norm1, self.w_q) # [T, D_MODEL]
        K = matmul_2d(x_norm1, self.w_k) # [T, D_MODEL]
        V = matmul_2d(x_norm1, self.w_v) # [T, D_MODEL]
        print(f"\n[Step 3] Q, K, V 投影 (Linear Projections):")
        print(f"         Q 形状: [{T}, {D_MODEL}] | K 形状: [{T}, {D_MODEL}] | V 形状: [{T}, {D_MODEL}]")
        print(f"         物理含义:")
        print(f"         - Query (Q): 当前位置想要'询问'什么？")
        print(f"         - Key   (K): 当前位置拥有什么'检索特征'？")
        print(f"         - Value (V): 当前位置真正携带的'内容信息'。")

        # -------------------------------------------------------------
        # Step 4: 多头注意力拆分与因果注意力打分
        # -------------------------------------------------------------
        print(f"\n[Step 4] 多头注意力计算与因果掩码 (Causal Attention):")
        # 针对每个 Head 独立计算
        head_outputs = []
        for h in range(N_HEADS):
            col_start = h * D_HEAD
            col_end = (h + 1) * D_HEAD
            # 提取属于头 h 的子矩阵 [T, D_HEAD]
            Q_h = [[row[c] for c in range(col_start, col_end)] for row in Q]
            K_h = [[row[c] for c in range(col_start, col_end)] for row in K]
            V_h = [[row[c] for c in range(col_start, col_end)] for row in V]

            # 计算注意力分数: Q_h * K_h^T / sqrt(D_HEAD)
            scale = math.sqrt(D_HEAD)
            attn_scores = [[0.0] * T for _ in range(T)]
            for i in range(T):
                for j in range(T):
                    dot = sum(Q_h[i][d] * K_h[j][d] for d in range(D_HEAD))
                    attn_scores[i][j] = dot / scale

            # 注入因果掩码 (Causal Masking): j > i 的未来位置设为 -inf
            for i in range(T):
                for j in range(T):
                    if j > i:
                        attn_scores[i][j] = -1e9 # 代表 -inf

            # Softmax 归一化
            attn_weights = [softmax(row) for row in attn_scores]

            if h == 0:
                print(f"         🔎 【头 0 的注意力热力权重矩阵 (Attention Map)】(行=当前词, 列=被关注词):")
                header = f"         {'':<6} | " + " | ".join([f"{tokens_str[c]:^6}" for c in range(T)])
                print(header)
                print("         " + "-" * len(header))
                for r in range(T):
                    row_str = f"         {tokens_str[r]:<6} | " + " | ".join([f"{attn_weights[r][c]*100:5.1f}%" for c in range(T)])
                    print(row_str)
                print(f"         👉 观察: 右上角全部为 0.0%，严格杜绝向后偷看未来！")

            # 加权求和 context = attn_weights * V_h -> [T, D_HEAD]
            context_h = matmul_2d(attn_weights, V_h)
            head_outputs.append(context_h)

        # -------------------------------------------------------------
        # Step 5: 多头拼接与 Output 投影
        # -------------------------------------------------------------
        concat_context = []
        for i in range(T):
            row = []
            for h in range(N_HEADS):
                row.extend(head_outputs[h][i])
            concat_context.append(row) # [T, D_MODEL]

        attn_out = matmul_2d(concat_context, self.w_o) # [T, D_MODEL]
        print(f"\n[Step 5] 多头特征拼接与输出线性映射 (Out Projection):")
        print(f"         多头拼接形状: [{T}, {D_MODEL}]  -->  W_o 投影输出: [{T}, {D_MODEL}]")

        # -------------------------------------------------------------
        # Step 6: 第 1 层残差连接
        # -------------------------------------------------------------
        x_residual1 = [[x[i][d] + attn_out[i][d] for d in range(D_MODEL)] for i in range(T)]
        print(f"\n[Step 6] 残差连接 1 (Residual Connection):")
        print(f"         公式: X_new = X + Attention(X)  -->  形状: [{T}, {D_MODEL}]")
        print(f"         说明: 保持深层网络信息畅通传递，防止表征退化。")

        # -------------------------------------------------------------
        # Step 7: 第 2 层 RMSNorm + SwiGLU 前馈网络 (MLP)
        # -------------------------------------------------------------
        x_norm2 = rms_norm(x_residual1, self.norm2_w)
        gate = matmul_2d(x_norm2, self.w_gate) # [T, HIDDEN_DIM]
        up   = matmul_2d(x_norm2, self.w_up)   # [T, HIDDEN_DIM]

        # SwiGLU: silu(gate) * up
        swiglu_act = []
        for i in range(T):
            swiglu_row = [silu(gate[i][k]) * up[i][k] for k in range(HIDDEN_DIM)]
            swiglu_act.append(swiglu_row)

        ffn_out = matmul_2d(swiglu_act, self.w_down) # [T, D_MODEL]
        x_residual2 = [[x_residual1[i][d] + ffn_out[i][d] for d in range(D_MODEL)] for i in range(T)]
        print(f"\n[Step 7] SwiGLU 门控前馈网络 (MLP) 与残差连接 2:")
        print(f"         升维中间层形状: [{T}, {HIDDEN_DIM}]  -->  降维输出形状: [{T}, {D_MODEL}]")
        print(f"         公式: X_final = X_res1 + (SiLU(X·W_gate) ⊙ (X·W_up)) · W_down")

        # -------------------------------------------------------------
        # Step 8: Final RMSNorm + LM Head 投影到词表
        # -------------------------------------------------------------
        x_final = rms_norm(x_residual2, self.final_norm_w)
        logits_all = matmul_2d(x_final, self.lm_head) # [T, VOCAB_SIZE]
        print(f"\n[Step 8] 最终归一化与 LM Head (输出投影):")
        print(f"         输入形状: [{T}, {D_MODEL}]  -->  输出 Logits 形状: [{T}, {VOCAB_SIZE}]")
        print(f"         说明: 每个位置都产生了一个大小为 {VOCAB_SIZE} 的未归一化打分向量！")

        # -------------------------------------------------------------
        # Step 9: 提取最后一个位置做预测与采样
        # -------------------------------------------------------------
        last_logits = logits_all[-1] # [VOCAB_SIZE]
        last_probs = softmax(last_logits)
        
        # 排序选出前 3 名最高可能性的词
        ranked_indices = sorted(range(VOCAB_SIZE), key=lambda idx: last_probs[idx], reverse=True)
        
        print(f"\n[Step 9] 预测下一个词 (针对最后一个 Token '{tokens_str[-1]}'):")
        print(f"         候选词概率分布排行榜 (Top 3):")
        for rank, idx in enumerate(ranked_indices[:3]):
            print(f"         🥇 第 {rank+1} 名: Token '{VOCAB[idx]}' (ID={idx:<2d}) | 原始 Logit: {last_logits[idx]:6.2f} | 概率: {last_probs[idx]*100:5.2f}%")

        best_token_id = ranked_indices[0]
        print(f"\n🎯 贪婪解码决策 (Greedy Choice): 采纳 Token '{VOCAB[best_token_id]}' (ID={best_token_id})")
        return best_token_id


def run_micro_trace():
    tracer = MicroLlamaTracer()

    # Prefill 阶段: 输入初始 Prompt "大" "模" "型"
    prompt_ids = [WORD2ID["大"], WORD2ID["模"], WORD2ID["型"]]
    next_token_1 = tracer.forward_trace(prompt_ids, phase_name="Prefill (首字全量计算)")

    # Decode 阶段 1: 将新生成的 token 拼接入序列，预测后续词
    full_context_step1 = prompt_ids + [next_token_1]
    next_token_2 = tracer.forward_trace(full_context_step1, phase_name="Decode Step 1 (生成第 2 个字)")

    print("\n" + "=" * 76)
    print("🎉 完整生成结果追踪:")
    all_tokens = prompt_ids + [next_token_1, next_token_2]
    result_text = "".join([VOCAB[tid] for tid in all_tokens])
    print(f"   输入提示词 : {''.join([VOCAB[tid] for tid in prompt_ids])}")
    print(f"   模型输出   : {result_text}")
    print("=" * 76)


if __name__ == "__main__":
    run_micro_trace()
