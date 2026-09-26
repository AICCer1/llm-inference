// =====================================================================
// softmax_kernels.cu —— 逐行 softmax 的三种 CUDA 实现
//
// 配套：第 9 篇（Online Softmax 的严格推导与 IO 复杂度）、第 8 篇（Roofline）
//       lab09_flash_attention/flash_attention_numpy.py（同一套算法的 NumPy 版）
//
// 三个 kernel 的访存量（每行 N 个 float，4 字节）：
//   1. softmax_3pass_kernel   3 读 + 1 写 = 16N 字节   ← 朴素写法，lab09 §1
//   2. softmax_fused_kernel   2 读 + 1 写 = 12N 字节   ← online softmax 合成一遍
//   3. softmax_shmem_kernel   1 读 + 1 写 =  8N 字节   ← 行常驻 shared memory，理论上界
//
// 用 lab08 在你的 5060 Ti 上实测的 380 GB/s，三者应该有 1 : 1.33 : 2 的耗时比。
// 这就是第 8 篇说的：优化的对象是字节，不是 FLOPs。
// =====================================================================

#define NEG_INF __int_as_float(0xff800000)   // -inf，设备端常量，不需要 <math.h>

// ---------------------------------------------------------------------
// FlashAttention 的那个"合并"操作，作用在 (m, s) 这一对上
//   m = max(m1, m2)
//   s = s1*exp(m1-m) + s2*exp(m2-m)
// 它是第 9 篇在线 softmax 归纳法的核心：块与块之间可以任意顺序合并。
// ---------------------------------------------------------------------
__device__ __forceinline__ void online_combine(float& m, float& s, float m2, float s2) {
    float m_new = fmaxf(m, m2);
    s = s * __expf(m - m_new) + s2 * __expf(m2 - m_new);
    m = m_new;
}

// =====================================================================
// 版本 1：朴素三段式（3 读 + 1 写）
// 三个 kernel，中间结果 m[]、sum[] 落到显存。这就是 PyTorch 早期
// 以及 lab09 NumPy 版做的事，也是 FlashAttention 要解决的"反复搬运"。
// =====================================================================
extern "C" __global__ void rowmax_kernel(const float* __restrict__ x,
                                         float* __restrict__ m, int N) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const float* __restrict__ r = x + (long long)row * N;

    float mx = NEG_INF;
    for (int i = threadIdx.x; i < N; i += blockDim.x) mx = fmaxf(mx, r[i]);

    red[threadIdx.x] = mx;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    if (threadIdx.x == 0) m[row] = red[0];
}

extern "C" __global__ void rowsum_kernel(const float* __restrict__ x,
                                         const float* __restrict__ m,
                                         float* __restrict__ s, int N) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const float* __restrict__ r = x + (long long)row * N;
    const float mx = m[row];

    float acc = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) acc += __expf(r[i] - mx);

    red[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) s[row] = red[0];
}

extern "C" __global__ void normalize_kernel(const float* __restrict__ x,
                                            const float* __restrict__ m,
                                            const float* __restrict__ s,
                                            float* __restrict__ y, int N) {
    const int row = blockIdx.x;
    const float* __restrict__ r = x + (long long)row * N;
    float* __restrict__ o = y + (long long)row * N;
    const float mx = m[row];
    const float inv = 1.f / s[row];

    for (int i = threadIdx.x; i < N; i += blockDim.x) o[i] = __expf(r[i] - mx) * inv;
}

// =====================================================================
// 版本 2：单 kernel online softmax（2 读 + 1 写）
// 维护"运行中的 (m, s)"，一次扫描同时算出行最大值和归一化因子，
// 省掉了中间数组和一次完整读。但要写回时得再读一遍 x。
// =====================================================================
extern "C" __global__ void softmax_fused_kernel(const float* __restrict__ x,
                                                float* __restrict__ y, int N) {
    extern __shared__ float red[];              // 2 * blockDim.x：前半存 m，后半存 s
    float* sm = red;
    float* ss = red + blockDim.x;

    const int row = blockIdx.x;
    const float* __restrict__ r = x + (long long)row * N;
    float* __restrict__ o = y + (long long)row * N;

    // ---- 一趟扫描，边扫边修正 ----
    float m = NEG_INF, s = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        online_combine(m, s, r[i], 1.f);        // 每个元素自身是 (m2=v, s2=1)

    // ---- block 内用同一套合并规则把每线程的 (m,s) 归约 ----
    sm[threadIdx.x] = m;
    ss[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            float m2 = sm[threadIdx.x + off], s2 = ss[threadIdx.x + off];
            online_combine(sm[threadIdx.x], ss[threadIdx.x], m2, s2);
        }
        __syncthreads();
    }
    const float gm = sm[0];
    const float gs = ss[0];                     // 全部同步过，直接广播

    // ---- 第二遍读 x 写结果 ----
    const float inv = 1.f / gs;
    for (int i = threadIdx.x; i < N; i += blockDim.x) o[i] = __expf(r[i] - gm) * inv;
}

// =====================================================================
// 版本 3：行常驻 shared memory（1 读 + 1 写 = 带宽下界）
// 和版本 2 的唯一区别：第一遍顺手把行缓存到 shared，第二遍从片上读。
// 这一版的耗时应该正好等于"读写一个张量"的时间 —— Roofline 的下界。
// 前提是 N*4 + 2*blockDim*4 装得进 shared memory。
// =====================================================================
extern "C" __global__ void softmax_shmem_kernel(const float* __restrict__ x,
                                                float* __restrict__ y, int N) {
    extern __shared__ float sh[];
    float* row_buf = sh;                        // N 个 float
    float* sm = sh + N;                         // blockDim.x
    float* ss = sm + blockDim.x;                // blockDim.x

    const int row = blockIdx.x;
    const float* __restrict__ r = x + (long long)row * N;
    float* __restrict__ o = y + (long long)row * N;

    float m = NEG_INF, s = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = r[i];
        row_buf[i] = v;                         // 唯一的 HBM 读
        online_combine(m, s, v, 1.f);
    }

    sm[threadIdx.x] = m;
    ss[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            float m2 = sm[threadIdx.x + off], s2 = ss[threadIdx.x + off];
            online_combine(sm[threadIdx.x], ss[threadIdx.x], m2, s2);
        }
        __syncthreads();
    }
    const float gm = sm[0];
    const float inv = 1.f / ss[0];

    for (int i = threadIdx.x; i < N; i += blockDim.x) o[i] = __expf(row_buf[i] - gm) * inv;
}
