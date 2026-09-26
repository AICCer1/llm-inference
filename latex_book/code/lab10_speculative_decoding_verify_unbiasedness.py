"""
Lab 10 - 验证脚本: 用蒙特卡洛实验"亲眼看到"投机采样的无偏性定理与期望加速比公式 (零外部依赖)

对应教程: zero_to_hero_tutorial/10_投机解码：无偏性证明与期望加速比.md

实验清单:
  1. 单 token：重复 20 万次"草稿采样 → 接受/拒绝 → 残差重采样"，经验分布 = 目标分布 p（而不是草稿分布 q）
  2. 草稿再差也无偏：q 与 p 几乎相反时，输出分布仍然是 p，只是接受率很低；实测接受率 = Σ min(p, q)
  3. 两种常见错误写法会产生偏差：(a) 拒绝后直接从 p 重采样 (b) 拒绝后从 q 重采样
  4. 多 token：K 个草稿一轮验证，实测每轮产出 = (1 - α^(K+1)) / (1 - α)；扫描最优 K 与加速比
"""

import random

N_SAMPLES = 200_000


def sample(dist, rng):
    r = rng.random()
    c = 0.0
    for i, p in enumerate(dist):
        c += p
        if r < c:
            return i
    return len(dist) - 1


def normalize(v):
    s = sum(v)
    return [x / s for x in v]


def tv(a, b):
    """总变差距离 TV(a, b) = 0.5 Σ |a - b|"""
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def speculative_step(p, q, rng, mode="correct"):
    """返回 (输出 token, 是否接受了草稿)"""
    x = sample(q, rng)
    if rng.random() < min(1.0, p[x] / q[x]):
        return x, True
    if mode == "correct":
        residual = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
        return sample(normalize(residual), rng), False
    if mode == "resample_p":        # 常见错误 1：拒绝后直接从目标分布 p 采样
        return sample(p, rng), False
    if mode == "resample_q":        # 常见错误 2：拒绝后再从草稿分布 q 采样
        return sample(q, rng), False
    raise ValueError(mode)


def empirical(p, q, mode, seed):
    rng = random.Random(seed)
    counts = [0] * len(p)
    acc = 0
    for _ in range(N_SAMPLES):
        x, a = speculative_step(p, q, rng, mode)
        counts[x] += 1
        acc += a
    return [c / N_SAMPLES for c in counts], acc / N_SAMPLES


def show(name, dist):
    print(f"   {name:<22}: " + " ".join(f"{x:.3f}" for x in dist))


def title(s):
    print("\n" + "=" * 78)
    print(f"[实验] {s}")
    print("=" * 78)


def exp1():
    title(f"实验 1: 单 token 投机采样的输出分布（重复 {N_SAMPLES:,} 次）")
    p = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]      # 目标（大）模型
    q = [0.20, 0.35, 0.10, 0.20, 0.05, 0.05, 0.05]      # 草稿（小）模型
    emp, acc = empirical(p, q, "correct", 0)
    show("目标分布 p", p)
    show("草稿分布 q", q)
    show("投机采样实测输出", emp)
    alpha = sum(min(a, b) for a, b in zip(p, q))
    print(f"   TV(实测, p) = {tv(emp, p):.4f}   TV(实测, q) = {tv(emp, q):.4f}")
    assert tv(emp, p) < 0.01 and abs(acc - alpha) < 0.01, '投机采样输出分布应等于 p'
    print(f"   接受率: 实测 {acc:.4f} | 理论 Σmin(p,q) = {alpha:.4f} = 1 - TV(p,q) = {1 - tv(p, q):.4f}")
    print("   --> 输出分布贴合 p（TV 只剩约 1/√N 量级的采样噪声），与 q 明显不同 —— 定理成立。")


def exp2():
    title("实验 2: 草稿模型再差，结果依然无偏（只是慢）")
    p = [0.70, 0.20, 0.05, 0.03, 0.02]
    q = [0.02, 0.03, 0.05, 0.20, 0.70]                   # 几乎与 p 相反
    emp, acc = empirical(p, q, "correct", 1)
    show("目标分布 p", p)
    show("草稿分布 q（很烂）", q)
    show("投机采样实测输出", emp)
    print(f"   TV(实测, p) = {tv(emp, p):.4f}；接受率实测 {acc:.3f}，理论 {sum(min(a, b) for a, b in zip(p, q)):.3f}")
    assert tv(emp, p) < 0.01
    print("   --> 草稿的质量只影响速度（接受率），不影响正确性。")


def exp3():
    title("实验 3: 两种常见的错误实现 —— 分布出现系统性偏差")
    p = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
    q = [0.20, 0.35, 0.10, 0.20, 0.05, 0.05, 0.05]
    show("目标分布 p", p)
    for mode, name in [("correct", "[OK] 残差分布重采样"), ("resample_p", "[X] 拒绝后从 p 采样"), ("resample_q", "[X] 拒绝后从 q 采样")]:
        emp, _ = empirical(p, q, mode, 2)
        show(name, emp)
        print(f"   {'':<22}  TV(实测, p) = {tv(emp, p):.4f}")
        assert (tv(emp, p) < 0.01) == (mode == 'correct'), f'{mode} 的偏差不符合预期'
    print("   --> '拒绝后从 p 采样'看起来很合理，其实会让 q 偏高的 token 被双重计入。")
    print("      只有第 10 篇推导出的残差分布 max(0, p-q)/Z 能恰好补齐缺口。")


def exp4():
    title("实验 4: K 个草稿一轮 —— 期望产出公式与最优 K")
    # 为了让每个位置的接受率相同（公式的前提），这里每个位置都使用同一对 (p, q)
    p = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
    q = normalize([0.6 * a + 0.4 / len(p) for a in p])   # 一个"还凑合"的草稿模型
    alpha = sum(min(a, b) for a, b in zip(p, q))
    rng = random.Random(3)
    c = 0.1                                              # 草稿模型单步耗时 = 大模型的 10%
    print(f"   每个位置接受率 α = {alpha:.3f}，草稿/目标耗时比 c = {c}")
    print(f"   {'K':>3} | {'实测每轮产出':>12} | {'公式 (1-α^(K+1))/(1-α)':>24} | {'加速比 E/(Kc+1)':>16}")
    best = (0, 0)
    for K in [1, 2, 3, 4, 5, 6, 8, 12, 16]:
        rounds, produced = 20_000, 0
        for _ in range(rounds):
            n = 0
            for _ in range(K):
                x, ok = speculative_step(p, q, rng)
                n += 1
                if not ok:
                    break
            else:
                n += 1                                   # 全部接受 → 免费多采一个
            produced += n
        e_emp = produced / rounds
        e_th = (1 - alpha ** (K + 1)) / (1 - alpha)
        sp = e_th / (K * c + 1)
        best = max(best, (sp, K))
        print(f"   {K:>3} | {e_emp:12.3f} | {e_th:24.3f} | {sp:16.2f}x")
        assert abs(e_emp - e_th) / e_th < 0.03, '实测每轮产出应与公式吻合'
    print(f"   --> 实测与公式吻合。K 太大时 α^i 衰减、草稿成本却线性增加，最优 K ≈ {best[1]}（加速 {best[0]:.2f}x）。")
    print("      注意这里假设'大模型验证 K 个 token 与验证 1 个一样快'——只在小 batch 的访存受限区成立（第 8 篇）。")


if __name__ == "__main__":
    exp1()
    exp2()
    exp3()
    exp4()
