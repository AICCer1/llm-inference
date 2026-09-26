"""
Lab 03: 从零手写自动求导引擎 (Autograd)，并用数值差分验证第 3 篇的每一个公式 (零外部依赖)

对应教程: zero_to_hero_tutorial/03_神经网络如何学习：导数、链式法则、反向传播与交叉熵.md
思路来自 Karpathy 的 micrograd (https://github.com/karpathy/micrograd)，这里精简并加上了中文注释与验证实验。

实验清单:
  1. 复现第 3 篇"手算表格"：w=3, x=2, b=1, y=10 时的梯度 dL/dw = -12, dL/db = -6
  2. 梯度检验：自动求导 vs 中心差分数值导数（验证你写的反向传播是对的）
  3. 验证 Softmax 雅可比 p_i(δ_ij - p_j) 与 "Softmax+交叉熵梯度 = p - y"
  4. 验证矩阵乘反向传播公式 dL/dW = X^T G, dL/dX = G W^T
  5. 梯度消失：多层 Sigmoid 连乘 vs 加上残差连接
  6. 历史重演：单层感知机学不会 XOR，加一层隐藏层 + 非线性后学会了
"""

import math
import random


# =====================================================================
# 核心：一个标量节点 Value，记录"值 + 梯度 + 它是怎么算出来的"
# =====================================================================
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                 # dL/d(本节点)，反向传播时累加
        self._backward = lambda: None   # 把本节点的梯度分发给输入节点的函数
        self._prev = set(_children)
        self._op = _op

    # ---------- 基本运算：每个运算都要写"局部导数 × 上游梯度" ----------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1。用 += 是因为一个节点可能被多条路径使用（多路径相加）
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad   # (e^x)' = e^x
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad  # 上界 1/4
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---------- 反向传播：拓扑排序后，从输出往输入逐个调用 _backward ----------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0  # dL/dL = 1
        for v in reversed(topo):
            v._backward()

    # ---------- 语法糖 ----------
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)
    def __rtruediv__(self, other): return other * self ** -1
    def __repr__(self): return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"


def title(s):
    print("\n" + "=" * 74)
    print(f"🔬 {s}")
    print("=" * 74)


def numeric_grad(f, xs, i, h=1e-5):
    """中心差分: (f(x+h) - f(x-h)) / 2h，只用来"检验"，不用来训练"""
    xp = list(xs); xp[i] += h
    xm = list(xs); xm[i] -= h
    return (f(xp) - f(xm)) / (2 * h)


# =====================================================================
# 实验 1: 复现第 3 篇手算表格
# =====================================================================
def exp1_hand_table():
    title("实验 1: 复现第 3 篇的手算反向传播表格")
    x, w, b, y = Value(2.0), Value(3.0), Value(1.0), Value(10.0)
    u = w * x
    y_hat = u + b
    e = y_hat - y
    L = e ** 2
    L.backward()
    print(f"   前向: u = {u.data}, y_hat = {y_hat.data}, e = {e.data}, L = {L.data}")
    print(f"   反向: dL/de = {e.grad}, dL/dy_hat = {y_hat.grad}, dL/du = {u.grad}")
    print(f"         dL/dw = {w.grad}   (表格中手算: -12)")
    print(f"         dL/db = {b.grad}   (表格中手算:  -6)")
    assert (w.grad, b.grad) == (-12.0, -6.0)
    lr = 0.01
    print(f"   一步梯度下降 (η={lr}): w: 3 → {3 - lr * w.grad:.2f}，b: 1 → {1 - lr * b.grad:.2f}，"
          f"新预测 = {(3 - lr * w.grad) * 2 + (1 - lr * b.grad):.2f}（向真实值 10 靠近）")


# =====================================================================
# 实验 2: 梯度检验
# =====================================================================
def exp2_gradcheck():
    title("实验 2: 梯度检验 —— 自动求导 vs 数值差分")

    def f_value(v):
        a, b, c = v
        return ((a * b).tanh() + (c ** 2) * a.exp() - (b / c).log() * 0.5).sigmoid()

    def f_float(xs):
        a, b, c = xs
        return 1 / (1 + math.exp(-(math.tanh(a * b) + c ** 2 * math.exp(a) - 0.5 * math.log(b / c))))

    xs = [0.3, 1.7, 0.9]
    vs = [Value(x) for x in xs]
    out = f_value(vs)
    out.backward()
    print("   f(a,b,c) = sigmoid( tanh(a·b) + c²·e^a − 0.5·ln(b/c) )")
    for i, name in enumerate("abc"):
        ng = numeric_grad(f_float, xs, i)
        rel = abs(vs[i].grad - ng) / max(1e-12, abs(ng))
        print(f"   ∂f/∂{name}: 自动求导 = {vs[i].grad:+.8f} | 数值差分 = {ng:+.8f} | 相对误差 = {rel:.1e}")
        assert rel < 1e-6, f"∂f/∂{name} 梯度检验失败"
    print("   👉 相对误差远小于 1e-6：说明每个算子的 _backward 都写对了。以后你写任何新算子，都应该这样检验。")


# =====================================================================
# 实验 3: Softmax 雅可比 & Softmax + 交叉熵
# =====================================================================
def softmax_values(zs):
    m = max(z.data for z in zs)                 # 平移不变性，防溢出（常数不参与求导）
    exps = [(z - m).exp() for z in zs]
    s = sum(exps, Value(0.0))
    return [e / s for e in exps]


def exp3_softmax_ce():
    title("实验 3: Softmax 雅可比 p_i(δ_ij − p_j) 与交叉熵梯度 p − y")
    z_raw = [2.0, 1.0, 0.1, -1.5]

    # (a) 对 p_0 反向传播，得到雅可比矩阵的第 0 行 ∂p_0/∂z_j
    zs = [Value(z) for z in z_raw]
    ps = softmax_values(zs)
    ps[0].backward()
    p = [q.data for q in ps]
    print(f"   z = {z_raw}")
    print(f"   p = softmax(z) = {[round(q, 4) for q in p]}")
    print("   雅可比第 0 行 ∂p_0/∂z_j：")
    for j in range(len(z_raw)):
        formula = p[0] * ((1.0 if j == 0 else 0.0) - p[j])
        print(f"      j={j}: 自动求导 = {zs[j].grad:+.6f} | 公式 p_0(δ_0j − p_j) = {formula:+.6f}")
        assert abs(zs[j].grad - formula) < 1e-12

    # (b) 交叉熵：真实类别 y = 2
    y = 2
    zs = [Value(z) for z in z_raw]
    ps = softmax_values(zs)
    loss = -ps[y].log()
    loss.backward()
    print(f"\n   真实类别 y = {y}，交叉熵 L = −log p_y = {loss.data:.4f}")
    for j in range(len(z_raw)):
        formula = p[j] - (1.0 if j == y else 0.0)
        print(f"      ∂L/∂z_{j}: 自动求导 = {zs[j].grad:+.6f} | 公式 p_j − 1[j=y] = {formula:+.6f}")
        assert abs(zs[j].grad - formula) < 1e-12
    print("   👉 梯度 = 预测分布 − 真实分布。正确类别的 logit 被推高，其余被压低，力度正比于它们各自的概率。")

    # (c) 饱和现象：logits 放大 10 倍后，Softmax 雅可比的数值全体变小
    zs = [Value(z * 10) for z in z_raw]
    ps = softmax_values(zs)
    ps[0].backward()
    print(f"\n   把 logits 放大 10 倍（模拟没有除以 √d_k）后，∂p_0/∂z_j = {[f'{z.grad:+.2e}' for z in zs]}")
    print("   👉 p_0 ≈ 1，雅可比整体坍缩到接近 0 —— 这就是第 4 篇所说的 Softmax 饱和导致梯度消失。")


# =====================================================================
# 实验 4: 矩阵乘的反向传播公式
# =====================================================================
def exp4_matmul_backward():
    title("实验 4: 验证矩阵乘反向传播公式 dL/dW = Xᵀ G, dL/dX = G Wᵀ")
    rng = random.Random(0)
    B, K, N = 2, 3, 2
    X = [[Value(rng.uniform(-1, 1)) for _ in range(K)] for _ in range(B)]
    W = [[Value(rng.uniform(-1, 1)) for _ in range(N)] for _ in range(K)]
    Y = [[sum((X[i][k] * W[k][j] for k in range(K)), Value(0.0)) for j in range(N)] for i in range(B)]
    # 随便定义一个标量损失: L = Σ c_ij * Y_ij ，于是 G = ∂L/∂Y = c
    C = [[rng.uniform(-1, 1) for _ in range(N)] for _ in range(B)]
    L = sum((Y[i][j] * C[i][j] for i in range(B) for j in range(N)), Value(0.0))
    L.backward()

    G = C
    dW_formula = [[sum(X[i][k].data * G[i][j] for i in range(B)) for j in range(N)] for k in range(K)]
    dX_formula = [[sum(G[i][j] * W[k][j].data for j in range(N)) for k in range(K)] for i in range(B)]
    err_w = max(abs(W[k][j].grad - dW_formula[k][j]) for k in range(K) for j in range(N))
    err_x = max(abs(X[i][k].grad - dX_formula[i][k]) for i in range(B) for k in range(K))
    print(f"   X: [{B},{K}], W: [{K},{N}], Y = XW: [{B},{N}]")
    print(f"   dL/dW 自动求导 vs Xᵀ G 最大误差: {err_w:.1e}")
    print(f"   dL/dX 自动求导 vs G Wᵀ 最大误差: {err_x:.1e}")
    assert err_w < 1e-12 and err_x < 1e-12
    print("   👉 反向传播需要两个矩阵乘（各与前向同样大小），所以反向 ≈ 2 倍前向 FLOPs，训练 ≈ 6N、推理 ≈ 2N。")


# =====================================================================
# 实验 5: 梯度消失 & 残差
# =====================================================================
def exp5_vanishing():
    depth = 10
    title(f"实验 5: 梯度消失 —— {depth} 层 Sigmoid 连乘 vs 加上残差连接")

    x = Value(0.5)
    h = x
    for _ in range(depth):
        h = (h * 1.0).sigmoid()
    h.backward()
    print(f"   纯 Sigmoid 堆 {depth} 层: dOut/dx = {x.grad:.3e}   (每层导数 ≤ 1/4，(1/4)^{depth} ≈ {0.25 ** depth:.1e})")
    assert x.grad <= 0.25 ** depth, "Sigmoid 连乘的梯度不应超过 (1/4)^depth"

    x = Value(0.5)
    h = x
    for _ in range(depth):
        h = h + (h * 1.0).sigmoid()  # 残差：x + F(x)
    h.backward()
    print(f"   加残差后 {depth} 层:       dOut/dx = {x.grad:.3e}   (每层导数 = 1 + F'(x) ≥ 1，梯度一路畅通)")
    assert x.grad >= 1.0, "有残差时梯度不应小于 1"


# =====================================================================
# 实验 6: 历史重演 —— XOR
# =====================================================================
def train_xor(hidden):
    rng = random.Random(42)
    data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]

    if hidden == 0:
        # 单层：y = sigmoid(w·x + b)，只能画一条直线
        W1 = [Value(rng.uniform(-1, 1)) for _ in range(2)]
        b1 = Value(0.0)
        params = W1 + [b1]

        def forward(x):
            return (W1[0] * x[0] + W1[1] * x[1] + b1).sigmoid()
    else:
        W1 = [[Value(rng.uniform(-1, 1)) for _ in range(hidden)] for _ in range(2)]
        b1 = [Value(0.0) for _ in range(hidden)]
        W2 = [Value(rng.uniform(-1, 1)) for _ in range(hidden)]
        b2 = Value(0.0)
        params = [w for row in W1 for w in row] + b1 + W2 + [b2]

        def forward(x):
            hs = [(W1[0][j] * x[0] + W1[1][j] * x[1] + b1[j]).tanh() for j in range(hidden)]
            return (sum((hs[j] * W2[j] for j in range(hidden)), Value(0.0)) + b2).sigmoid()

    lr = 0.5
    for step in range(2000):
        # 二分类交叉熵: -[y log p + (1-y) log(1-p)]
        loss = Value(0.0)
        for x, y in data:
            p = forward(x)
            loss = loss - (p.log() * y + (1 - p).log() * (1 - y))
        loss = loss / len(data)
        for q in params:
            q.grad = 0.0              # ⚠️ 梯度是累加的，每步必须清零（PyTorch 里的 optimizer.zero_grad()）
        loss.backward()
        for q in params:
            q.data -= lr * q.grad
    preds = [forward(x).data for x, _ in data]
    return loss.data, preds


def exp6_xor():
    title("实验 6: 历史重演 —— 1969 年感知机学不会 XOR，1986 年多层网络 + 反向传播学会了")
    for hidden in [0, 4]:
        loss, preds = train_xor(hidden)
        if hidden == 0:
            assert abs(loss - 0.6931) < 0.01, "单层模型在 XOR 上应停在 ln2"
        else:
            assert loss < 0.05 and [round(p) for p in preds] == [0, 1, 1, 0], "两层网络应学会 XOR"
        name = "单层 (无隐藏层)" if hidden == 0 else f"两层 (隐藏层 {hidden} 个 tanh 神经元)"
        print(f"   {name:<28}: 训练后损失 = {loss:.4f} | 对 (00,01,10,11) 的预测 = {[round(p, 3) for p in preds]} (目标 0,1,1,0)")
    print("   👉 单层模型损失卡在 ln2 ≈ 0.693（等于瞎猜）：一条直线分不开 XOR。")
    print("      加一层非线性隐藏层后损失趋近 0 —— 这就是第 1 篇说的'激活函数让网络能折出曲面'。")


if __name__ == "__main__":
    exp1_hand_table()
    exp2_gradcheck()
    exp3_softmax_ce()
    exp4_matmul_backward()
    exp5_vanishing()
    exp6_xor()
