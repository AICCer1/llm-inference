"""
Lab 06: KV Cache 原理与基准测试仿真 (零外部依赖，自动适配 Pure Python / NumPy / PyTorch)

本脚本直观演示：
1. 朴素自回归解码（无 KV Cache）：每次重新计算前文所有 token 的 K, V 投影。
2. 缓存自回归解码（有 KV Cache）：每次只计算新 token 的 Q, K, V，并追加到缓存中。
3. 验证两种方式输出完全等价，并对比推理耗时与计算量差异。
"""

import time
import math
import random

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


# ==========================================
# 矩阵运算轻量封装 (优先 NumPy，无环境时回退纯 Python)
# ==========================================
class MatrixOps:
    @staticmethod
    def random_matrix(rows, cols, seed=42):
        if HAVE_NUMPY:
            np.random.seed(seed)
            return np.random.randn(rows, cols) * 0.05
        else:
            rng = random.Random(seed)
            return [[rng.gauss(0, 0.05) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def matmul(A, B):
        if HAVE_NUMPY:
            return np.dot(A, B)
        else:
            n, m = len(A), len(A[0])
            p = len(B[0])
            res = [[0.0] * p for _ in range(n)]
            for i in range(n):
                for k in range(m):
                    aik = A[i][k]
                    for j in range(p):
                        res[i][j] += aik * B[k][j]
            return res

    @staticmethod
    def transpose(A):
        if HAVE_NUMPY:
            return A.T
        else:
            return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

    @staticmethod
    def concat_rows(A, B):
        if HAVE_NUMPY:
            return np.concatenate([A, B], axis=0)
        else:
            return [row[:] for row in A] + [row[:] for row in B]

    @staticmethod
    def softmax_row(row):
        max_val = max(row) if not HAVE_NUMPY else float(np.max(row))
        if HAVE_NUMPY:
            exp_vals = np.exp(row - max_val)
            return exp_vals / np.sum(exp_vals)
        else:
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            return [v / sum_exp for v in exp_vals]

    @staticmethod
    def max_diff(A, B):
        if HAVE_NUMPY:
            return float(np.max(np.abs(A - B)))
        else:
            diff = 0.0
            for i in range(len(A)):
                for j in range(len(A[0])):
                    diff = max(diff, abs(A[i][j] - B[i][j]))
            return diff


class MinimalAttention:
    """自注意力机制：分别实现朴素全量计算与 KV Cache 增量计算"""
    def __init__(self, d_model=128):
        self.d_model = d_model
        self.w_q = MatrixOps.random_matrix(d_model, d_model, seed=10)
        self.w_k = MatrixOps.random_matrix(d_model, d_model, seed=20)
        self.w_v = MatrixOps.random_matrix(d_model, d_model, seed=30)
        self.w_o = MatrixOps.random_matrix(d_model, d_model, seed=40)

    def forward_naive(self, seq_tokens):
        """
        无 KV Cache: 输入包含前缀在内的完整历史 [seq_len, d_model]
        仅取最后一个 token 的预测结果
        """
        # 全量重算全部历史 token 的 Q, K, V
        Q = MatrixOps.matmul(seq_tokens, self.w_q)
        K = MatrixOps.matmul(seq_tokens, self.w_k)
        V = MatrixOps.matmul(seq_tokens, self.w_v)
        
        # 只提取最后一个位置的查询向量 q_last: [1, d_model]
        q_last = Q[-1:]
        
        # 计算注意力分数: q_last 与所有历史 token 的 K 转置相乘 -> [1, seq_len]
        K_T = MatrixOps.transpose(K)
        scores_mat = MatrixOps.matmul(q_last, K_T)
        
        scale = math.sqrt(self.d_model)
        if HAVE_NUMPY:
            scores = scores_mat / scale
            attn_weights = MatrixOps.softmax_row(scores[0])
            context = np.dot(attn_weights.reshape(1, -1), V)
        else:
            scores = [v / scale for v in scores_mat[0]]
            attn_weights = MatrixOps.softmax_row(scores)
            context = MatrixOps.matmul([attn_weights], V)
            
        output = MatrixOps.matmul(context, self.w_o)
        return output

    def forward_with_cache(self, new_token, kv_cache=None):
        """
        有 KV Cache:
        - new_token: [1, d_model] 仅输入当前步最新生成的这 1 个 token
        - kv_cache: (K_past, V_past) 保存的历史 K 和 V
        """
        # 仅对新 token 计算单向量的 q, k, v
        q_new = MatrixOps.matmul(new_token, self.w_q)
        k_new = MatrixOps.matmul(new_token, self.w_k)
        v_new = MatrixOps.matmul(new_token, self.w_v)
        
        # 拼接更新缓存
        if kv_cache is None or kv_cache[0] is None:
            k_all = k_new
            v_all = v_new
        else:
            k_past, v_past = kv_cache
            k_all = MatrixOps.concat_rows(k_past, k_new)
            v_all = MatrixOps.concat_rows(v_past, v_new)
            
        new_kv_cache = (k_all, v_all)
        
        # 注意力计算: q_new (1, d_model) 与 k_all (seq_len, d_model)
        K_T = MatrixOps.transpose(k_all)
        scores_mat = MatrixOps.matmul(q_new, K_T)
        
        scale = math.sqrt(self.d_model)
        if HAVE_NUMPY:
            scores = scores_mat / scale
            attn_weights = MatrixOps.softmax_row(scores[0])
            context = np.dot(attn_weights.reshape(1, -1), v_all)
        else:
            scores = [v / scale for v in scores_mat[0]]
            attn_weights = MatrixOps.softmax_row(scores)
            context = MatrixOps.matmul([attn_weights], v_all)
            
        output = MatrixOps.matmul(context, self.w_o)
        return output, new_kv_cache


def run_benchmark():
    engine_name = "NumPy 加速矩阵运算" if HAVE_NUMPY else "Python 原生标准库运算 (零外部依赖)"
    print("==================================================")
    print("🚀 KV Cache 原理与自回归基准测试")
    print(f"💻 计算引擎: {engine_name}")
    print("==================================================\n")

    d_model = 128
    prompt_len = 32
    decode_steps = 40
    
    layer = MinimalAttention(d_model=d_model)
    prompt = MatrixOps.random_matrix(prompt_len, d_model, seed=100)
    
    print(f"📌 测试配置: 隐藏层维度={d_model}, Prefill Prompt 长度={prompt_len}, Decode 生成步数={decode_steps}\n")

    # 1. 验证数学等价性
    print("🔍 阶段 1: 验证输出数值等价性...")
    kv_cache = None
    # Prefill: 灌入 Prompt 构建初始缓存
    for i in range(prompt_len):
        token_i = prompt[i:i+1]
        _, kv_cache = layer.forward_with_cache(token_i, kv_cache)
    
    # 模拟下一 token 输入
    next_input = MatrixOps.random_matrix(1, d_model, seed=200)
    
    # 无缓存方式预测
    full_seq = MatrixOps.concat_rows(prompt, next_input)
    out_naive = layer.forward_naive(full_seq)
    
    # 有缓存方式预测
    out_cache, _ = layer.forward_with_cache(next_input, kv_cache)
    
    diff = MatrixOps.max_diff(out_naive, out_cache)
    print(f"   最大绝对误差 (Max Absolute Diff): {diff:.8e}")
    assert diff < 1e-6, f"有/无 KV Cache 输出不一致，最大误差 {diff}"
    print("   ✅ 校验通过：有/无 KV Cache 输出结果数值完全一致！\n")

    # 2. 耗时基准测试：两条路径从同一个 prompt 出发、生成同一条序列，只对 Decode 部分计时
    print(f"⏱️ 阶段 2: 耗时测试 (Prefill 后自回归生成 {decode_steps} 步，两边都只计 Decode 耗时)...")

    # 无 Cache：每一步都把完整历史重新送进去
    tokens_history = [prompt[i:i+1] for i in range(prompt_len)]
    naive_outs, naive_times = [], []
    t0 = time.perf_counter()
    for step in range(decode_steps):
        st = time.perf_counter()
        seq = tokens_history[0]
        for t in tokens_history[1:]:
            seq = MatrixOps.concat_rows(seq, t)
        out = layer.forward_naive(seq)
        tokens_history.append(out)
        naive_outs.append(out)
        naive_times.append(time.perf_counter() - st)
    naive_total = time.perf_counter() - t0

    # 有 Cache：先 Prefill（不计时），Prefill 最后一个位置的输出就是第 1 个生成结果
    kv_cache, out = None, None
    for i in range(prompt_len):
        out, kv_cache = layer.forward_with_cache(prompt[i:i+1], kv_cache)
    cache_outs, cache_times = [out], [0.0]   # 第 1 步的输出来自 Prefill，Decode 耗时记为 0
    t0 = time.perf_counter()
    for step in range(1, decode_steps):
        st = time.perf_counter()
        out, kv_cache = layer.forward_with_cache(out, kv_cache)   # 只送入上一步生成的 1 个 token
        cache_outs.append(out)
        cache_times.append(time.perf_counter() - st)
    cache_total = time.perf_counter() - t0

    worst = max(MatrixOps.max_diff(a, b) for a, b in zip(naive_outs, cache_outs))
    assert worst < 1e-6, f"两条路径生成的序列不一致，最大误差 {worst}"
    print(f"   ✅ 两条路径逐步生成的 {decode_steps} 个输出全部一致（最大误差 {worst:.1e}）")
    print(f"   （注：朴素路径的第 1 步本身就是一次完整 Prefill，所以两边的计时范围略有不同，差距被略微高估）")

    # 打印对比结果
    print(f"\n📊 汇总结果:")
    print(f"   - 无 KV Cache 累计生成耗时 : {naive_total * 1000:.2f} ms")
    print(f"   - 有 KV Cache 累计生成耗时 : {cache_total * 1000:.2f} ms")
    speedup = naive_total / cache_total if cache_total > 0 else 1.0
    print(f"   - 🚀 加速比 (Speedup)      : {speedup:.2f}x\n")

    print(f"📈 步进耗时采样 (毫秒):")
    sample_indices = [0, decode_steps // 4, decode_steps // 2, decode_steps - 1]
    print(f"   {'步数 (Step)':<14} | {'无 Cache 耗时 (ms)':<22} | {'有 Cache 耗时 (ms)':<22}")
    print(f"   {'-'*14}-+-{'-'*22}-+-{'-'*22}")
    for idx in sample_indices:
        print(f"   Step {idx+1:<9} | {naive_times[idx]*1000:<22.3f} | {cache_times[idx]*1000:<22.3f}")

    print("\n💡 深入思考与核心结论:")
    print("   1. 【计算量差异】：无 Cache 情况下，每生成 1 个新 token，都需要重新计算前面所有 token 的 QKV 投影矩阵。")
    print("      在长文本场景下，计算复杂度高达 O(N^2)。有 Cache 后，QKV 投影只需计算当前这 1 个 token，复杂度降为 O(N)。")
    print("   2. 【显存代价】：虽然省去了重算，但所有历史 token 的 K 和 V 矩阵必须常驻显存。")
    print("      在高并发、多轮对话场景下，KV Cache 迅速吞噬显存，这正是 vLLM 提出 PagedAttention 与连续批处理要解决的痛点！")


if __name__ == "__main__":
    run_benchmark()
