"""
Lab 11 - 进阶: 用 NumPy 从零实现 SmoothQuant、AWQ、GPTQ，验证第 11 篇的每个论断

对应教程: zero_to_hero_tutorial/11_量化：从每位6dB到SmoothQuant、AWQ与GPTQ.md

实验清单:
  1. 每多 1 位，信噪比 +6 dB（均匀量化误差方差 = S²/12）
  2. 激活离群通道：per-tensor INT8 量化被几个通道毁掉
  3. SmoothQuant：严格的数学恒等变换 + 把激活的量化难度转移给权重（W8A8）
  4. AWQ：放大关键输入通道的权重，W4 分组量化的输出误差下降
  5. GPTQ：逐列量化 + 用 Hessian 逆补偿误差，对比直接四舍五入（RTN）

运行:
  .venv/bin/python lab11_quantization/advanced_quant_numpy.py
"""

import numpy as np

rng = np.random.default_rng(0)


def title(s):
    print("\n" + "=" * 78)
    print(f"[实验] {s}")
    print("=" * 78)


def snr_db(ref, approx):
    return 10 * np.log10((ref ** 2).mean() / ((ref - approx) ** 2).mean())


def quant_sym(x, bits, axis=None):
    """对称量化再反量化。axis=None: 整个张量一个 scale；axis=k: 沿第 k 维求 max，即每个"其它维度的切片"一个 scale"""
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(x).max() if axis is None else np.abs(x).max(axis=axis, keepdims=True)
    scale = np.maximum(amax, 1e-12) / qmax
    return np.clip(np.round(x / scale), -qmax, qmax) * scale


def quant_group(W, bits, group):
    """W: [C_in, C_out]，沿输入维度每 group 个元素一组（每个输出列各自分组），对称量化"""
    C_in, C_out = W.shape
    Wg = W.reshape(C_in // group, group, C_out)
    return quant_sym(Wg, bits, axis=1).reshape(C_in, C_out)


# =====================================================================
# 实验 1
# =====================================================================
def exp1_six_db():
    title("实验 1: 每多 1 位，信噪比约 +6 dB")
    x = rng.standard_normal(1_000_000)
    prev = None
    for b in range(2, 9):
        s = snr_db(x, quant_sym(x, b))
        delta = "" if prev is None else f"(+{s - prev:.2f} dB)"
        print(f"   INT{b}: SNR = {s:6.2f} dB {delta}")
        assert prev is None or 5.5 < s - prev < 8, '每位增益应在 6 dB 附近'
        prev = s
    print("   --> 位数 +1 → 格宽 S 减半 → 误差方差 S²/12 变为 1/4 → SNR +10·log10(4) ≈ 6.02 dB。\n"
          "      理论上 INT8→INT4 损失约 24 dB；但那是高分辨率近似，低位宽下反而损失更多 ——"
          "本机实测请见上面打印的实际 SQNR 差值（通常 25 dB 左右，低比特那几档每 bit 超过 6 dB）。")


# =====================================================================
# 共享的"模拟 LLM 线性层"数据
# =====================================================================
def make_layer(T=512, C_in=1024, C_out=1024, n_outlier=6, outlier_scale=40.0):
    X = rng.standard_normal((T, C_in))
    outlier_ch = rng.choice(C_in, n_outlier, replace=False)
    X[:, outlier_ch] *= outlier_scale        # 少数固定通道在每个 token 上都很大（LLM.int8 的发现）
    W = rng.standard_normal((C_in, C_out)) * 0.02
    return X, W, outlier_ch


# =====================================================================
# 实验 2 & 3
# =====================================================================
def exp2_3_smoothquant():
    X, W, outlier_ch = make_layer()
    Y = X @ W
    title("实验 2: 激活离群通道让 per-tensor 量化失效")
    ch_max = np.abs(X).max(axis=0)
    print(f"   {X.shape[1]} 个通道里有 {len(outlier_ch)} 个离群通道：离群通道最大值 ≈ {ch_max[outlier_ch].mean():.0f}，"
          f"普通通道 ≈ {np.median(ch_max):.1f}")
    normal = np.setdiff1d(np.arange(X.shape[1]), outlier_ch)
    Xq = quant_sym(X, 8)
    print(f"   per-tensor INT8 量化激活：整体 SNR = {snr_db(X, Xq):.1f} dB，但普通通道 SNR 只有 {snr_db(X[:, normal], Xq[:, normal]):.1f} dB")
    print(f"   （格宽由最大值约 {np.abs(X).max():.0f} 决定，普通通道的值大多被舍入到了 0 附近的几格里）")
    print(f"   权重分布规整：per-tensor INT8 量化权重 SNR = {snr_db(W, quant_sym(W, 8)):.1f} dB")

    title("实验 3: SmoothQuant —— 恒等变换 X·W = (X/s)·(s·W)，把难度从激活转移给权重")
    base = snr_db(Y, quant_sym(X, 8) @ quant_sym(W, 8))
    print(f"   直接 W8A8（均 per-tensor）: 输出 SNR = {base:.1f} dB")
    for alpha in [0.25, 0.5, 0.75]:
        s = np.abs(X).max(axis=0) ** alpha / np.abs(W).max(axis=1) ** (1 - alpha)
        Xs, Ws = X / s, W * s[:, None]
        exact = np.abs(Xs @ Ws - Y).max()
        out = snr_db(Y, quant_sym(Xs, 8) @ quant_sym(Ws, 8))
        print(f"   α = {alpha:<4}: 未量化时变换前后最大差 {exact:.1e}（严格等价）| W8A8 输出 SNR = {out:.1f} dB (+{out - base:.1f})")
        assert exact < 1e-10 and out > base, 'SmoothQuant 应严格等价且改善 W8A8'
    print("   --> α=0.5 附近效果最好：激活的离群值被压下去，权重相应变大一点，两边都好量化。")
    print("      推理时 1/s 合并进前一层 RMSNorm 的 γ、s 合并进 W，计算图零改动、零额外开销。")


# =====================================================================
# 实验 4: AWQ
# =====================================================================
def exp4_awq():
    title("实验 4: AWQ —— 按激活幅度放大关键权重通道，再做 W4 分组量化")
    T, C_in, C_out, group = 512, 1024, 512, 128
    X = rng.standard_normal((T, C_in))
    salient = rng.choice(C_in, C_in // 100, replace=False)   # 约 1% 的输入通道激活很大
    X[:, salient] *= 15.0
    W = rng.standard_normal((C_in, C_out)) * 0.02
    Y = X @ W

    rtn = snr_db(Y, X @ quant_group(W, 4, group))
    print(f"   W4 分组量化 (group={group})，直接四舍五入 RTN：输出 SNR = {rtn:.2f} dB")
    act_mag = np.abs(X).mean(axis=0)
    best = (rtn, 0.0)
    for beta in np.linspace(0, 1, 11):
        s = act_mag ** beta
        s = s / np.sqrt(s.max() * s.min())                     # 归一化，避免整体缩放
        Wq = quant_group(W * s[:, None], 4, group) / s[:, None]  # 等价于 Q(w·s)·(x/s)
        val = snr_db(Y, X @ Wq)
        best = max(best, (val, beta))
    print(f"   AWQ 网格搜索 β ∈ [0,1]，s = (激活平均幅度)^β：最优 β = {best[1]:.1f}，输出 SNR = {best[0]:.2f} dB (+{best[0] - rtn:.2f} dB)")
    assert best[0] > rtn, 'AWQ 应改善输出 SNR'
    s = act_mag ** best[1]; s = s / np.sqrt(s.max() * s.min())
    err_rtn = np.abs((W - quant_group(W, 4, group))[salient]).mean()
    err_awq = np.abs((W - quant_group(W * s[:, None], 4, group) / s[:, None])[salient]).mean()
    print(f"   关键通道上的权重误差：RTN {err_rtn:.2e} → AWQ {err_awq:.2e}（这些误差会被 15 倍的大激活放大，所以最要紧）")
    assert err_awq < err_rtn, 'AWQ 应减小关键通道的权重误差'
    print("   --> 放大 s 倍后，关键通道相对于格宽更'粗'，舍入误差相对变小；代价是同组其它权重精度略降，搜索 β 找平衡。")


# =====================================================================
# 实验 5: GPTQ
# =====================================================================
def gptq(W, X, bits=4, damp=0.01):
    """
    W: [C_out, C_in]（与 nn.Linear 相同），X: [C_in, n] 校准输入。最小化 ||WX - ŴX||²。
    每个输出行一个 scale（per-channel），按输入通道从左到右逐列量化，并用 H^{-1} 把误差摊给还没量化的列。
    """
    W = W.copy()
    C_out, C_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=1) / qmax                      # 用原始权重确定格宽，与 RTN 相同
    H = 2 * X @ X.T
    H += damp * np.mean(np.diag(H)) * np.eye(C_in)            # 阻尼：保证可逆、数值稳定
    Hinv = np.linalg.inv(H)
    for j in range(C_in):
        w = W[:, j]
        q = np.clip(np.round(w / scale), -qmax, qmax) * scale
        err = (w - q) / Hinv[j, j]
        W[:, j] = q
        W[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])        # OBS 最优补偿：让后面的列替第 j 列"还债"
        Hinv -= np.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]  # 从问题中移除第 j 列（高斯消元一步）
    return W


def exp5_gptq():
    title("实验 5: GPTQ vs 直接四舍五入 (RTN)，INT4 per-channel")
    C_in, C_out, n = 256, 256, 4096
    # 让输入通道之间"相关"：真实网络里的激活有很强的相关结构，这正是 GPTQ 能补偿误差的前提
    mix = rng.standard_normal((C_in, 32)) @ rng.standard_normal((32, C_in)) / 6 + np.eye(C_in)
    X_cal = mix @ rng.standard_normal((C_in, n))
    X_test = mix @ rng.standard_normal((C_in, n))              # 独立的测试数据，检验是否过拟合校准集
    W = rng.standard_normal((C_out, C_in)) * 0.02
    Y = W @ X_test

    qmax = 7
    scale = np.abs(W).max(axis=1, keepdims=True) / qmax
    W_rtn = np.clip(np.round(W / scale), -qmax, qmax) * scale
    W_gptq = gptq(W, X_cal)
    s_rtn, s_gptq = snr_db(Y, W_rtn @ X_test), snr_db(Y, W_gptq @ X_test)
    print(f"   权重本身的误差:   RTN SNR = {snr_db(W, W_rtn):.2f} dB | GPTQ SNR = {snr_db(W, W_gptq):.2f} dB")
    print(f"   层输出（测试集）: RTN SNR = {s_rtn:.2f} dB | GPTQ SNR = {s_gptq:.2f} dB (+{s_gptq - s_rtn:.2f} dB)")
    assert s_gptq > s_rtn + 5, 'GPTQ 应显著优于 RTN'
    print("   --> 有意思的是：GPTQ 的【权重】误差反而更大（它故意改动了还没量化的权重），但【输出】误差显著更小。")
    print("      GPTQ 优化的是 ||WX - ŴX||，不是 ||W - Ŵ||：输入通道相关时，一列的舍入误差可以由相关的列抵消。")

    X_ind = rng.standard_normal((C_in, n))
    W_g2 = gptq(W, X_ind)
    X_ind_test = rng.standard_normal((C_in, n))
    gain = snr_db(W @ X_ind_test, W_g2 @ X_ind_test) - snr_db(W @ X_ind_test, W_rtn @ X_ind_test)
    print(f"   对照：若输入通道彼此独立（H 近似对角），GPTQ 相对 RTN 只提升 {gain:+.2f} dB —— 没有相关性就没有东西可补偿。")


if __name__ == "__main__":
    exp1_six_db()
    exp2_3_smoothquant()
    exp4_awq()
    exp5_gptq()
