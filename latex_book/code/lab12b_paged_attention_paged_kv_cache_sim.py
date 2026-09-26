"""
Lab 12b: PagedAttention、写时复制、前缀缓存与 Chunked Prefill 仿真

对应教程: zero_to_hero_tutorial/12_推理服务系统：PagedAttention、前缀缓存与调度.md

实验清单:
  1. 块分配器 + 块表 + 分页注意力：KV 存在不连续的物理块里，逐块用 Online Softmax 计算，结果与连续存储完全一致
  2. 显存利用率：按 max_len 预留连续显存 vs 分页，在同样的显存预算下能同时服务多少请求
  3. 并行采样的写时复制（Copy-on-Write）：共享提示词块，只在写入共享块时才复制
  4. 前缀缓存（哈希式，类似 vLLM Automatic Prefix Caching）：共享系统提示词时省掉多少 Prefill
  5. Chunked Prefill：一个长 Prefill 插入时，正在 Decode 的请求会卡多久；切块后如何改善

运行（需要 NumPy）:
  .venv/bin/python lab12b_paged_attention/paged_kv_cache_sim.py
"""

import math
import random

import numpy as np

BLOCK = 16   # 每个物理块存 16 个 token 的 K/V（vLLM 默认值）


def title(s):
    print("\n" + "=" * 78)
    print(f"[实验] {s}")
    print("=" * 78)


# =====================================================================
# 核心数据结构
# =====================================================================
class BlockAllocator:
    """物理块池：空闲链表 + 引用计数（对应操作系统的物理页框管理）"""

    def __init__(self, num_blocks):
        self.free = list(range(num_blocks))
        self.ref = [0] * num_blocks
        self.num_blocks = num_blocks

    def alloc(self):
        if not self.free:
            raise MemoryError("没有空闲块 → 调度器需要抢占（换出或重算）某个请求")
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def incref(self, b):
        self.ref[b] += 1

    def decref(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)

    @property
    def used(self):
        return self.num_blocks - len(self.free)


class PagedKVCache:
    """
    物理存储：K_pool/V_pool 形状 [num_blocks, BLOCK, d]
    每个序列一张块表：block_table[seq_id] = [物理块号, ...]，第 i 个 token 在 (block_table[i // BLOCK], i % BLOCK)
    """

    def __init__(self, num_blocks, d):
        self.alloc = BlockAllocator(num_blocks)
        self.K = np.zeros((num_blocks, BLOCK, d))
        self.V = np.zeros((num_blocks, BLOCK, d))
        self.tables = {}
        self.lens = {}
        self.cow_copies = 0

    def new_seq(self, sid):
        self.tables[sid], self.lens[sid] = [], 0

    def fork(self, parent, child):
        """并行采样：子序列共享父序列的全部块（只加引用计数，不拷贝数据）"""
        self.tables[child] = list(self.tables[parent])
        self.lens[child] = self.lens[parent]
        for b in self.tables[child]:
            self.alloc.incref(b)

    def append(self, sid, k, v):
        table, n = self.tables[sid], self.lens[sid]
        if n % BLOCK == 0:                       # 上一块写满了（或还没有块）→ 分配新块
            table.append(self.alloc.alloc())
        else:
            last = table[-1]
            if self.alloc.ref[last] > 1:         # 最后一块被别人共享 → 写时复制
                new = self.alloc.alloc()
                self.K[new], self.V[new] = self.K[last], self.V[last]
                self.alloc.decref(last)
                table[-1] = new
                self.cow_copies += 1
        b = table[-1]
        self.K[b, n % BLOCK], self.V[b, n % BLOCK] = k, v
        self.lens[sid] = n + 1

    def free_seq(self, sid):
        for b in self.tables.pop(sid):
            self.alloc.decref(b)
        self.lens.pop(sid)

    def attention(self, sid, q):
        """分页注意力：按块表逐块读取 K/V，用 Online Softmax（第 9 篇）累加 —— 从不拼出连续的 K/V"""
        m, l, acc = -np.inf, 0.0, np.zeros_like(q)
        n = self.lens[sid]
        for i, b in enumerate(self.tables[sid]):
            cnt = min(BLOCK, n - i * BLOCK)
            s = self.K[b, :cnt] @ q / math.sqrt(q.shape[0])
            m_new = max(m, s.max())
            p = np.exp(s - m_new)
            corr = math.exp(m - m_new) if m > -np.inf else 0.0
            l = l * corr + p.sum()
            acc = acc * corr + p @ self.V[b, :cnt]
            m = m_new
        return acc / l


# =====================================================================
# 实验 1: 分页注意力的正确性
# =====================================================================
def exp1_correctness():
    title("实验 1: KV 分散在不连续的物理块里，注意力结果与连续存储完全一致")
    rng = np.random.default_rng(0)
    d = 64
    cache = PagedKVCache(num_blocks=64, d=d)
    random.seed(0)
    random.shuffle(cache.alloc.free)            # 打乱空闲链表，让分到的物理块号杂乱无章
    ks, vs = {}, {}
    for sid, n in [("A", 50), ("B", 37), ("C", 70)]:
        cache.new_seq(sid)
        ks[sid], vs[sid] = [], []
    for t in range(70):                          # 三个请求交替写入（模拟连续批处理）
        for sid, n in [("A", 50), ("B", 37), ("C", 70)]:
            if t < n:
                k, v = rng.standard_normal(d), rng.standard_normal(d)
                cache.append(sid, k, v)
                ks[sid].append(k); vs[sid].append(v)
    for sid in ["A", "B", "C"]:
        q = rng.standard_normal(d)
        K, V = np.array(ks[sid]), np.array(vs[sid])
        s = K @ q / math.sqrt(d)
        ref = np.exp(s - s.max()) @ V / np.exp(s - s.max()).sum()
        out = cache.attention(sid, q)
        print(f"   请求 {sid}: {cache.lens[sid]:2d} 个 token，块表 = {cache.tables[sid]}，与连续计算的最大误差 {np.abs(out - ref).max():.1e}")
        assert np.abs(out - ref).max() < 1e-12
    print(f"   共使用 {cache.alloc.used} 个物理块；每个请求只有最后一块可能没写满。")


# =====================================================================
# 实验 2: 显存利用率
# =====================================================================
def exp2_utilization():
    title("实验 2: 同样的显存，预留连续空间 vs 分页，能同时服务多少请求？")
    rng = random.Random(42)
    budget_tokens = 64 * 1024                    # KV 显存总共能放 64K 个 token
    max_len = 2048
    # 真实负载：提示词和输出长度都是长尾分布，大多数请求远小于 max_len
    reqs = [(int(rng.lognormvariate(5.5, 0.8)), int(rng.lognormvariate(5.0, 0.9))) for _ in range(2000)]
    reqs = [(min(p, 1500), min(o, max_len - min(p, 1500))) for p, o in reqs]
    avg_final = sum(p + o for p, o in reqs) / len(reqs)

    static_slots = budget_tokens // max_len
    # 分页：每个请求按"当前长度"占块，这里取它们生命周期中点的平均长度估计稳态占用
    avg_blocks_mid = sum(math.ceil((p + o / 2) / BLOCK) for p, o in reqs) / len(reqs)
    paged_slots = int(budget_tokens / BLOCK / avg_blocks_mid)
    real_tokens_mid = sum(p + o / 2 for p, o in reqs) / len(reqs)
    waste_paged = 1 - real_tokens_mid / (avg_blocks_mid * BLOCK)
    waste_static = 1 - real_tokens_mid / max_len

    print(f"   负载: 2000 个请求，平均最终长度 {avg_final:.0f} token（max_len = {max_len}）")
    print(f"   KV 显存预算: {budget_tokens:,} 个 token 的空间")
    print(f"   {'方案':<26} | {'可同时服务':>10} | {'显存浪费率':>10}")
    print(f"   {'按 max_len 预留连续空间':<22} | {static_slots:>10} | {waste_static:>9.0%}")
    print(f"   {'PagedAttention (块=16)':<24} | {paged_slots:>10} | {waste_paged:>9.1%}")
    # 分页的两个核心收益，逐条断言：
    assert paged_slots > static_slots, "分页必须能服务更多并发请求，否则这个方案没有意义"
    assert waste_paged < waste_static, "分页的浪费率必须低于按 max_len 预留"
    # 分页的浪费上界是每序列最后一块没写满（< 16 token），不是 0
    assert waste_paged < 16 / avg_final, \
        f"分页浪费率 {waste_paged:.2%} 超过了「每序列最后一块没写满」的上界 {16 / avg_final:.2%}，" \
        f"说明块分配器有别的浪费"
    print(f"   --> 并发数提升 {paged_slots / static_slots:.1f} 倍。第 8 篇说过 Decode 时 batch 越大吞吐越高，")
    print("      所以省下的显存会直接变成吞吐。vLLM 论文实测旧系统只有 20%~38% 的 KV 显存真正存了 token。")


# =====================================================================
# 实验 3: 写时复制
# =====================================================================
def exp3_copy_on_write():
    title("实验 3: 并行采样 n=4 —— 引用计数共享提示词块 + 写时复制")
    rng = np.random.default_rng(1)
    d, prompt_len, gen_len, n = 8, 100, 30, 4
    cache = PagedKVCache(num_blocks=256, d=d)
    cache.new_seq("p")
    for _ in range(prompt_len):
        cache.append("p", rng.standard_normal(d), rng.standard_normal(d))
    print(f"   提示词 {prompt_len} token 写入后占 {cache.alloc.used} 块（最后一块只写了 {prompt_len % BLOCK}/{BLOCK}）")
    for i in range(n):
        cache.fork("p", f"s{i}")
    cache.free_seq("p")
    print(f"   fork 出 {n} 个子序列后仍只占 {cache.alloc.used} 块，每块引用计数 = {cache.alloc.ref[cache.tables['s0'][0]]}")
    for _ in range(gen_len):
        for i in range(n):
            cache.append(f"s{i}", rng.standard_normal(d), rng.standard_normal(d))
    no_share = n * math.ceil((prompt_len + gen_len) / BLOCK)
    print(f"   各自生成 {gen_len} token 后：共享方案用了 {cache.alloc.used} 块，发生写时复制 {cache.cow_copies} 次")
    assert cache.cow_copies == n - 1 and cache.alloc.used < no_share, '应恰好发生 n-1 次写时复制'
    print(f"   不共享时需要 {no_share} 块 → 节省 {1 - cache.alloc.used / no_share:.0%}")
    print(f"   --> 只有那个'被共享且没写满'的最后一块会被复制，而且只复制 {n - 1} 次而不是 {n} 次：")
    print(f"      前 {n - 1} 个子序列各复制一份后，原块的引用计数降到 1，最后一个子序列可以直接原地写入。")
    print("      前面写满的提示词块始终只存一份。")


# =====================================================================
# 实验 4: 前缀缓存
# =====================================================================
def exp4_prefix_cache():
    title("实验 4: 前缀缓存 —— 所有请求共享同一段系统提示词")
    rng = random.Random(7)
    system_prompt = [rng.randrange(50000) for _ in range(1000)]   # 1000 token 的系统提示词
    cached = {}                                                    # 哈希(前缀的所有 token) → 物理块号

    def prefill_with_cache(tokens):
        """逐个"写满的块"算哈希：键 = 从开头到这一块末尾的全部 token（不能只看本块，否则位置不同的相同内容会误命中）"""
        hit = 0
        for i in range(len(tokens) // BLOCK):
            key = hash(tuple(tokens[: (i + 1) * BLOCK]))
            if key in cached:
                hit += BLOCK
            else:
                break   # 前缀一旦不同，后面全部不能复用
        for i in range(hit // BLOCK, len(tokens) // BLOCK):
            cached[hash(tuple(tokens[: (i + 1) * BLOCK]))] = len(cached)
        return hit

    total, saved = 0, 0
    for r in range(20):
        user = [rng.randrange(50000) for _ in range(rng.randint(20, 200))]
        tokens = system_prompt + user
        hit = prefill_with_cache(tokens)
        total += len(tokens)
        saved += hit
        if r < 3:
            print(f"   请求 {r}: 共 {len(tokens)} token，命中缓存 {hit} token，只需 Prefill {len(tokens) - hit} token")
    print(f"   ... 20 个请求合计 {total} token，其中 {saved} token（{saved / total:.0%}）直接复用缓存，免去 Prefill")

    # 只改开头第 1 个 token（比如提示词里写入了当前时间）。
    # 必须【确定性地】换成一个不同的值：写成 rng.randrange(50000) 的话有 1/50000 的概率
    # 抽到和原来相同，于是缓存照样命中，而打印出来的结论就变成了假的。
    changed_first = (system_prompt[0] + 1) % 50000
    assert changed_first != system_prompt[0]
    hit = prefill_with_cache([changed_first] + system_prompt[1:] + [1, 2, 3])
    assert hit == 0, f"第 1 个 token 变了之后不该有任何前缀命中，实际命中 {hit} token"
    print(f"   [注意] 把系统提示词的第 1 个 token 改掉后：命中 {hit} token —— 整个前缀缓存全部失效")


# =====================================================================
# 实验 5: Chunked Prefill
# =====================================================================
def exp5_chunked_prefill():
    title("实验 5: Chunked Prefill —— 长 Prefill 插队时，正在 Decode 的用户会卡多久？")
    # 用第 8 篇的耗时模型，参数取自本机实测（RTX 5060 Ti）
    # [注意] 这两个数都是 measure_gpu_roofline.py **已经实测达到**的值，不要再乘 MFU：
    #    带宽 380 GB/s（实测拷贝带宽）、算力 48 TFLOP/s（实测 BF16 Tensor 稠密算力）。
    #    两侧必须同口径——只给算力降额、访存却用 achieved 值，比值会偏一倍。
    # 模型：Qwen2.5-1.5B (BF16 权重约 3.1 GB)
    W_bytes, n_params = 3.1e9, 1.54e9
    bw, peak, overhead = 380e9, 48e12, 3e-3              # 实测带宽 / 实测算力 + 3ms 固定开销

    def step_time(n_tokens):
        return max(W_bytes / bw, 2 * n_params * n_tokens / peak) + overhead

    decode_batch = 32
    long_prompt = 6000
    print(f"   场景: {decode_batch} 个用户正在 Decode，此时来了一个 {long_prompt} token 的长提示词")
    print(f"   单步耗时模型: max(读权重 {W_bytes / bw * 1e3:.1f} ms, 计算 2N×tokens/算力) + 固定开销 {overhead * 1e3:.0f} ms")
    print(f"   纯 Decode 一步（{decode_batch} token）: {step_time(decode_batch) * 1e3:.1f} ms")
    print(f"\n   {'方案':<22} | {'Prefill 分几步':>12} | {'Decode 用户最长卡顿':>18} | {'新请求 TTFT':>12}")
    for chunk in [None, 2048, 512, 256]:
        if chunk is None:
            steps = [long_prompt]
            name = "不切块（一次做完）"
        else:
            steps = [min(chunk, long_prompt - i) for i in range(0, long_prompt, chunk)]
            name = f"Chunked Prefill {chunk}"
        times = [step_time(s + decode_batch) for s in steps]
        print(f"   {name:<20} | {len(steps):>12} | {max(times) * 1e3:>15.1f} ms | {sum(times) * 1e3:>9.1f} ms")
    print("   --> 不切块时，32 个用户在这一步里全部卡住几百毫秒（TPOT 尖刺）。切成小块后，每步耗时有上界，")
    print("      代价是新请求的 TTFT 略增。块越小，Decode 越平滑，但 Prefill 越慢（每步都要重新读一遍权重）。")
    ridge_tokens = W_bytes / bw * peak / (2 * n_params)
    # 交叉核对：这里的拐点必须和 lab08 实测的 AI* 一致（都是 实测算力 / 实测带宽）
    ai_star = peak / bw
    assert 100 < ai_star < 160, \
        f"AI* = {ai_star:.0f} 与 lab08 在 5060 Ti 上实测的约 130 不符，检查两侧口径是否同源"
    print(f"      另外：AI* = 算力/带宽 = {ai_star:.0f} FLOP/Byte，与 lab08 实测的约 130 一致。")
    print(f"      所以这个模型每步 token 数低于约 {ridge_tokens:.0f} 时仍是访存受限（第 8 篇的拐点），")
    print(f"      此时往 Decode 批次里塞进一小段 Prefill 几乎不增加耗时 —— 这就是'两种瓶颈互补'。")
    print("      真实引擎（vLLM 的 max_num_batched_tokens）通常取几百到几千，在平滑度和 Prefill 效率间折中。")


if __name__ == "__main__":
    exp1_correctness()
    exp2_utilization()
    exp3_copy_on_write()
    exp4_prefix_cache()
    exp5_chunked_prefill()
