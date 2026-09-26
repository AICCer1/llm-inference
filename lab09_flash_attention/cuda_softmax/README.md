# Lab 09 附录：真机 CUDA —— 把 Online Softmax 写成 kernel

> 对应理论：[第 9 篇 §2~§3](../../zero_to_hero_tutorial/09_FlashAttention：OnlineSoftmax的严格推导与IO复杂度.md)、[第 8 篇（Roofline）](../../zero_to_hero_tutorial/08_硬件视角：GPU存储层级、FLOPs、算术强度与浮点格式.md)
> 前置：读完第 8、9 篇，跑过 [Lab 08 的 roofline 实测](../../lab08_memory_and_roofline/)
> 依赖：PyTorch（只为拿它自带的 NVRTC，**不需要 CUDA toolkit，也不需要 nvcc**）

主实验 [`flash_attention_numpy.py`](../flash_attention_numpy.py) 是 NumPy 仿真：它证明**数学**是对的，但所有"访存量"都是算出来的，不是量出来的。
这个附录补上最后一环——**同样的 online softmax，写成真正的 CUDA C++，在你的显卡上量出带宽**。

## 运行

```bash
.venv/bin/python lab09_flash_attention/cuda_softmax/run.py                        # 4096 x 4096
.venv/bin/python lab09_flash_attention/cuda_softmax/run.py --rows 8192 --cols 8192
```

⚠️ **张量必须明显大于 L2**（本机 32 MB），否则数据被整个缓存住，"DRAM 带宽"这个口径就失效了，会算出超过硬件峰值的假数字。脚本会自己检测并警告。

## 三个 kernel

同一件事（逐行 softmax）的三种写法，全部用第 9 篇的 `online_combine` 合并规则：

| 文件里的名字 | 写法 | 真正过 DRAM 的字节 |
| :--- | :--- | :--- |
| `softmax_3pass_kernel` | 三个独立 kernel：求 max → 求 sum → 归一化（= 主实验 §1 的朴素版） | 3 读 + 1 写 |
| `softmax_fused_kernel` | 单 kernel，边扫边维护 $(m, s)$，写完再读一遍 x | 1 读 + 1 写（第二遍命中 L2） |
| `softmax_shmem_kernel` | 同上，但第一遍顺手把整行缓存进 shared memory | 1 读 + 1 写（第二遍读片上） |

`online_combine` 就是 FlashAttention 分块合并的那一步：

```
m = max(m1, m2)
s = s1·exp(m1−m) + s2·exp(m2−m)
```

它满足结合律，所以既可以用来做 block 内归约，也可以用来合并不同块的 partial 结果——主实验 §4 的 FlashDecoding split-K 合并用的是同一个函数。

## 本机（RTX 5060 Ti, 36 SM, L2 32 MB）实测

> ⏱️ 计时类数字是本机某一次运行的**量级示例**（5 次预热 + 30 次 CUDA event 计时取平均），GPU 降频或有其它负载时会变。请看趋势和比例。

4096 × 4096（64 MB 张量）：

| 实现 | DRAM 搬运 | 耗时 | DRAM 量 | DRAM 带宽 | 占理论上界 | blocks/SM | 相对 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 段式 | 4 次 | 0.694 ms | 268 MB | 384~387 GB/s | 51% | 6 | 1.00x |
| fused | 2 次 (+1 片上) | 0.359 ms | 134 MB | 369~374 GB/s | **97~98%** | 6 | **1.9x** |
| shmem | 2 次 (+1 片上) | 0.360 ms | 134 MB | 369~373 GB/s | **97~98%** | 5 | **1.9x** |
| `torch.softmax` | — | 0.352 ms | — | — | — | — | — |

三个实现的最大绝对误差都是 **5.96e-08**（float32 的 eps）。shmem 版和 `torch.softmax` 的差距是 1.02x——**这个 kernel 已经是生产质量**。

### 这张表要读出来的东西

1. **三个实现都跑在 ~375 GB/s**，也就是 Lab 08 在这张卡上实测的 380 GB/s。它们**全都贴着 Roofline**，没有一个是"没优化好"。
2. 差别**只在搬运了多少字节**：268 MB vs 134 MB → 约 1.9x。这正是第 8 篇说的：在这个区域，优化的对象是字节，不是 FLOPs。
3. 三段式花的时间约是理论上界的 **2 倍**——**这就是 FlashAttention 要消灭的浪费**，只不过在注意力里被消灭的是 $N^2$ 的中间矩阵。
4. fused 版的第二遍读由 **L2** 兜住了（同一 block 刚读过自己的行，16 KB 还在 L2 里），所以它不需要 shared memory 也追平了 shmem 版。**先量再优化**——如果一上来就照搬"缓存到 shared"的教条，你会为一个不存在的瓶颈写代码。

### 一个反直觉的结果：shmem 版在大 N 时反而崩了

把尺寸换成 8192 × 8192：

| 实现 | 耗时 | DRAM 带宽 | 占上界 | blocks/SM |
| :--- | ---: | ---: | ---: | ---: |
| 3 段式 | 2.720 ms | 394.7 GB/s | 52% | 6 |
| fused | 1.438 ms | 373.5 GB/s | **97~98%** | 6 |
| shmem | 1.904 ms | **282.0 GB/s** | **74%** | **2** |

shmem 版掉到 74%，比 `torch.softmax` 还慢 1.34x。原因不是访存变多了，而是**占用率**：

- 行缓存要 `8192×4 + 2×256×4 = 34 KB` shared/block → 每个 SM 只能常驻 **2** 个 block
- fused 版只要 2 KB → 每个 SM 能常驻 **6** 个 block

占用率不够就掩盖不住访存延迟，带宽自然上不去。**这恰恰是 FlashAttention 不对整行做缓存、而要做分块的原因**：块小到能常驻，占用率才不掉，才能真正贴着带宽跑。

> 这个结论不是推导出来的，是脚本里那条 `assert` 在 N=8192 时**炸了**之后查出来的。脚本原文：
> `AssertionError: softmax_shmem_kernel 的 DRAM 带宽 282 GB/s 偏离实测带宽 380 GB/s 太多`
>
> 如果只 print 不断言，你只会看到一个"74%"和一个"98%"并排躺着，很容易被当成噪声划过去。

## 关于工具链（为什么这里没有 nvcc）

这个目录里的 `.cu` 是**标准 CUDA C++**：`__global__`、`__shared__`、`__syncthreads()`、动态 shared memory、block 级归约，一个不少。但它不是用 `nvcc` 编的，而是用 **NVRTC**（运行时编译）——PyTorch 的 `torch.cuda.jiterator` 等 JIT 路径走的就是这套机制，不少推理框架的运行时 kernel 编译也用它。

原因：apt 源里的 `nvidia-cuda-toolkit` 是 **CUDA 12.0，不支持 sm_120**，装了也编不出 RTX 50 系能跑的 cubin。而 PyTorch（cu128）自带 NVRTC 12.8，它支持。**你不需要为这个实验装任何东西。**

如果你以后装了 CUDA toolkit，同一份源码可以直接：

```bash
nvcc -arch=sm_120 -O3 -o softmax softmax_kernels.cu
```

`run.py` 里还踩了一个 cuda-python 特有的坑，值得记下来——`cuLaunchKernel` 的 `kernelParams` 不是常见的指针数组，而是 **`(值元组, ctypes 类型元组)`** 的两元组：

```python
params = ((d_ptr, n), (ctypes.c_void_p, ctypes.c_int))
driver.cuLaunchKernel(f, *grid, *block, shared_bytes, driver.CUstream(0), params, 0)
```

## 动手练习

1. **加 float4 向量化读写**，看带宽能不能超过 97~98%。然后用 Roofline 解释你的结果。
   （提示：先算清楚这个 kernel 的算术强度——它离拐点有多远？如果已经在带宽上界，向量化还能不能帮上忙？）
2. **把 `online_combine` 换成直接的两遍扫描**（先 `rowmax` 再 `rowsum`，都在一个 kernel 里、行缓存在 shared）。行为和 shmem 版一样，但没有 online 修正。测速度，并解释为什么第 9 篇还要用 online 形式。
3. **把 shmem 版改成真正的分块**（不缓存整行，只缓存 `BLOCK` 个元素，循环推进）。在 8192×8192 上跑，验证占用率恢复后带宽回到 97~98%。
   **做完这一题你就理解了 FlashAttention 的 tiling 是怎么来的。**
4. **把 `softmax_3pass_kernel` 的中间结果 `m[]`、`s[]` 改成 atomic 累加**，观察和三个独立 kernel 的差别——这就是"算子融合"省掉的一次完整读写。
5. **加一个 bf16 版本**（或 fp16），对比 float32 的带宽。想想为什么带宽不会翻倍（提示：第 8 篇的 β 是字节带宽，不是元素带宽）。

## 与主实验的关系

| | 主实验（NumPy） | 本附录（CUDA） |
| :--- | :--- | :--- |
| 证明什么 | 算法**数学**正确、合并满足结合律 | 同样的算法在**真机**上的带宽与占用率行为 |
| 访存量 | 按公式**算**出来 | 用 CUDA event **量**出来 |
| 能发现 | 复杂度量级 | L2 命中、占用率塌陷、离理论上界还差多少 |
| 局限 | 没有硬件 | 只做了 softmax，没做完整的注意力 |

看完这个再回头看第 9 篇 §3 那句"FlashAttention 的 FLOPs 与朴素完全相同，快是因为少搬了数据"——你手上有一张自己测出来的表可以验证它。
