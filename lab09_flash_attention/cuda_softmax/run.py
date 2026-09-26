#!/usr/bin/env python3
"""
用 NVRTC + CUDA Driver API 编译并运行 softmax_kernels.cu

为什么要绕 NVRTC 而不是 nvcc：
  这台机器没装 CUDA toolkit，而 apt 里的 nvidia-cuda-toolkit 是 CUDA 12.0，
  不支持 sm_120（RTX 50 系）。venv 里的 PyTorch 带了 CUDA 12.8 的 NVRTC，
  它能编译 sm_120，而且是 PyTorch JIT / vLLM 自定义 kernel 用的同一套机制。
  源码本身是标准 CUDA C++，装了 toolkit 之后 `nvcc -arch=sm_120` 直接能编。

用法：
    .venv/bin/python lab09_flash_attention/cuda_softmax/run.py              # 4096 行 x 4096 列
    .venv/bin/python lab09_flash_attention/cuda_softmax/run.py --rows 8192 --cols 2048

CUDA Driver API 的参数约定（cuda-python 12.9）：
    kernelParams 必须是 (值元组, ctypes 类型元组) 这种两元组，
    例如 ((d_ptr, n), (ctypes.c_void_p, ctypes.c_int))。
"""
import argparse
import ctypes
import os
import time

import numpy as np
from cuda.bindings import driver, nvrtc

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "softmax_kernels.cu")

BLOCK = 256          # 每行一个 block，blockDim.x = 256
MEASURED_BW = 380.0  # GB/s，lab08/measure_gpu_roofline.py 在这张卡上的实测值


# ---------------------------------------------------------------- NVRTC
def compile_cubin(src_path, arch="sm_120"):
    with open(src_path, "rb") as f:
        src = f.read()

    err, prog = nvrtc.nvrtcCreateProgram(src, b"softmax_kernels.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram: {err}")

    opts = [
        f"--gpu-architecture={arch}".encode(),
        b"--std=c++17",
        b"-default-device",
        b"--use_fast_math",
    ]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log_buf = bytearray(max(int(log_size), 1))
    nvrtc.nvrtcGetProgramLog(prog, log_buf)
    log = bytes(log_buf).decode(errors="replace").rstrip("\x00")

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCompileProgram failed:\n{log}")
    if log.strip():
        print("--- NVRTC log ---")
        print(log)

    _, size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin_buf = bytearray(int(size))
    nvrtc.nvrtcGetCUBIN(prog, cubin_buf)
    return bytes(cubin_buf)


# ---------------------------------------------------------------- 驱动封装
def ck(ret, what=""):
    """cuda-bindings 的调用统一返回 (err, ...)，这里只校验 err。"""
    err = ret[0] if isinstance(ret, tuple) else ret
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {err}")
    return ret


class Buf:
    def __init__(self, nbytes):
        _, self.ptr = driver.cuMemAlloc(nbytes)
        self.nbytes = nbytes

    def free(self):
        driver.cuMemFree(self.ptr)

    @property
    def addr(self):
        return int(self.ptr)


def pack(*args):
    """按 cuda-python 的约定打包 kernel 实参：(值元组, ctypes 类型元组)。"""
    vals, types = [], []
    for a in args:
        if isinstance(a, Buf):
            vals.append(a.addr)
            types.append(ctypes.c_void_p)
        elif isinstance(a, int):
            vals.append(a)
            types.append(ctypes.c_int)
        else:
            vals.append(float(a))
            types.append(ctypes.c_float)
    return (tuple(vals), tuple(types))


def timeit(fn, warmup=5, iters=30):
    """CUDA event 计时，返回每次调用的平均毫秒。"""
    for _ in range(warmup):
        fn()
    ck(driver.cuCtxSynchronize())

    _, start = driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT)
    _, stop = driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT)

    ck(driver.cuEventRecord(start, driver.CUstream(0)))
    for _ in range(iters):
        fn()
    ck(driver.cuEventRecord(stop, driver.CUstream(0)))
    ck(driver.cuEventSynchronize(stop))

    _, ms = driver.cuEventElapsedTime(start, stop)
    driver.cuEventDestroy(start)
    driver.cuEventDestroy(stop)
    return ms / iters


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=4096, help="每行长度 N")
    args = ap.parse_args()
    R, N = args.rows, args.cols

    print("=" * 78)
    print(f"逐行 Softmax 的三个 CUDA 实现 —— {R} 行 x {N} 列 float32")
    print("=" * 78)

    # ---- 初始化 ----
    ck(driver.cuInit(0), "cuInit")
    _, dev = driver.cuDeviceGet(0)
    _, raw_name = driver.cuDeviceGetName(128, dev)
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    name = name.rstrip("\x00").strip()
    ck(driver.cuCtxCreate(0, dev), "cuCtxCreate")

    _, sm_count = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
    _, max_shared = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev)
    _, l2_bytes = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev)
    print(f"设备 : {name}   SM={sm_count}   最大动态 shared/block={max_shared / 1024:.0f} KB   "
          f"L2={l2_bytes / 1024 / 1024:.0f} MB")

    # 前提检查：整个张量必须明显大于 L2，否则数据被缓存住，
    # "DRAM 带宽"这个口径就失效了（会算出远超硬件峰值的假数字）。
    tensor_mb = R * N * 4 / 1e6
    l2_mb = l2_bytes / 1024 / 1024
    roofline_ok = tensor_mb >= 2 * l2_mb
    if not roofline_ok:
        print(f"\n⚠️  张量只有 {tensor_mb:.0f} MB，不到 L2（{l2_mb:.0f} MB）的两倍。")
        print(f"    数据能整个放进缓存，此时【无法】用 DRAM 带宽和 Roofline 来解释耗时，")
        print(f"    算出来的\"带宽\"会超过硬件峰值。请用更大的尺寸，例如 --rows 4096 --cols 4096。")
        print(f"    下面仍然会跑并校验正确性，但不再做 Roofline 断言。\n")

    need_shared = N * 4 + 2 * BLOCK * 4          # 版本 3：行缓存 + 两个归约数组
    fit = need_shared <= max_shared
    print(f"shared 需求（版本 3）: {need_shared / 1024:.1f} KB  -> "
          f"{'可运行' if fit else '超出上限，跳过'}")

    # ---- 编译 ----
    print(f"\n编译 {os.path.basename(SRC)} ...")
    t0 = time.perf_counter()
    cubin = compile_cubin(SRC)
    print(f"  NVRTC 完成: {len(cubin):,} 字节 cubin，{time.perf_counter() - t0:.2f} s")

    _, mod = driver.cuModuleLoadData(cubin)

    def get(n):
        err, f = driver.cuModuleGetFunction(mod, n.encode())
        ck(err, f"cuModuleGetFunction({n})")
        return f

    K = {n: get(n) for n in (
        "rowmax_kernel", "rowsum_kernel", "normalize_kernel",
        "softmax_fused_kernel", "softmax_shmem_kernel")}

    # ---- 数据 ----
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((R, N), dtype=np.float32) * 2.0).astype(np.float32)

    d_x, d_y = Buf(x.nbytes), Buf(x.nbytes)
    d_m, d_s = Buf(R * 4), Buf(R * 4)
    ck(driver.cuMemcpyHtoD(d_x.ptr, x.ctypes.data, x.nbytes), "H2D")

    grid, block = (R, 1, 1), (BLOCK, 1, 1)

    def launch(f, shmem, params):
        err, = driver.cuLaunchKernel(f, *grid, *block, shmem,
                                     driver.CUstream(0), params, 0)
        ck(err, "cuLaunchKernel")

    def fetch_y():
        out = np.empty_like(x)
        ck(driver.cuMemcpyDtoH(out.ctypes.data, d_y.ptr, out.nbytes), "D2H")
        return out

    # ---- 正确性 ----
    print("\n【正确性】与 numpy 参考实现比对（参考实现用 float64 计算）")
    ref = np.exp(x.astype(np.float64) - x.max(axis=1, keepdims=True, ).astype(np.float64))
    ref /= ref.sum(axis=1, keepdims=True)
    ref = ref.astype(np.float32)

    results = {}

    def check(tag, dram_tx, onchip_tx, thresh=2e-5):
        got = fetch_y()
        err = float(np.abs(got - ref).max())
        ok = err < thresh if not np.isnan(err) else True
        results[tag] = dict(dram=dram_tx, onchip=onchip_tx, err=err, ok=ok)
        mark = "✅" if ok else "❌"
        print(f"  {mark} {tag:<24} 最大绝对误差 = {err:.3e}")
        return ok

    # 版本 1：三段式（3 个 kernel，每个都从 DRAM 完整读一遍 x）
    launch(K["rowmax_kernel"], BLOCK * 4, pack(d_x, d_m, N))
    launch(K["rowsum_kernel"], BLOCK * 4, pack(d_x, d_m, d_s, N))
    launch(K["normalize_kernel"], 0, pack(d_x, d_m, d_s, d_y, N))
    ck(driver.cuCtxSynchronize())
    check("softmax_3pass_kernel", dram_tx=4, onchip_tx=0)   # 3 读 + 1 写，全在 DRAM

    # 版本 2：单 kernel online，同一 block 内连着读两遍自己的行（第二遍命中 L2）
    launch(K["softmax_fused_kernel"], 2 * BLOCK * 4, pack(d_x, d_y, N))
    ck(driver.cuCtxSynchronize())
    check("softmax_fused_kernel", dram_tx=2, onchip_tx=1)   # 1 读 + 1 写

    # 版本 3：行常驻 shared，第二遍从片上读
    if fit:
        launch(K["softmax_shmem_kernel"], need_shared, pack(d_x, d_y, N))
        ck(driver.cuCtxSynchronize())
        check("softmax_shmem_kernel", dram_tx=2, onchip_tx=1)

    if not all(r["ok"] for r in results.values()):
        raise SystemExit("❌ 有实现与参考结果不符，见上表")

    # ---- 性能 ----
    print("\n【性能】CUDA event 计时（5 次预热 + 30 次平均）")
    tensor_bytes = R * N * 4

    def run_3pass():
        launch(K["rowmax_kernel"], BLOCK * 4, pack(d_x, d_m, N))
        launch(K["rowsum_kernel"], BLOCK * 4, pack(d_x, d_m, d_s, N))
        launch(K["normalize_kernel"], 0, pack(d_x, d_m, d_s, d_y, N))

    def run_fused():
        launch(K["softmax_fused_kernel"], 2 * BLOCK * 4, pack(d_x, d_y, N))

    def run_shmem():
        launch(K["softmax_shmem_kernel"], need_shared, pack(d_x, d_y, N))

    timing = {
        "softmax_3pass_kernel": timeit(run_3pass),
        "softmax_fused_kernel": timeit(run_fused),
        "softmax_shmem_kernel": timeit(run_shmem) if fit else float("nan"),
    }

    def occupancy(f, block, shmem):
        err, n = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(f, block, shmem)
        return int(n)

    # 三段式里 rowmax 的配置最有代表性（rowsum 同形；normalize 不用 shared）
    occ_fn = {"softmax_3pass_kernel": (K["rowmax_kernel"], BLOCK * 4),
              "softmax_fused_kernel": (K["softmax_fused_kernel"], 2 * BLOCK * 4),
              "softmax_shmem_kernel": (K["softmax_shmem_kernel"], need_shared)}
    occ = {k: occupancy(f, BLOCK, sm) for k, (f, sm) in occ_fn.items()}

    hdr = (f"  {'实现':<24}{'DRAM 搬运':<14}{'耗时':>10}{'DRAM 量':>10}"
           f"{'DRAM 带宽':>11}{'占上界':>9}{'blocks/SM':>11}{'相对':>8}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    base = timing["softmax_3pass_kernel"]
    ideal_ms = tensor_bytes * 2 / (MEASURED_BW * 1e9) * 1e3
    roofline_frac = {}
    for k, ms in timing.items():
        if np.isnan(ms):
            continue
        r = results[k]
        dram_bytes = tensor_bytes * r["dram"]
        bw = dram_bytes / (ms * 1e-3) / 1e9
        roofline_frac[k] = ideal_ms / ms
        desc = f"{r['dram']} 次" + (f" (+{r['onchip']} 片上)" if r["onchip"] else "")
        print(f"  {k:<24}{desc:<14}{ms:>7.3f} ms{dram_bytes / 1e6:>7.0f} MB"
              f"{bw:>8.1f} GB/s{ideal_ms / ms * 100:>8.0f}%{occ[k]:>11}{base / ms:>7.2f}x")
    print("  （DRAM 带宽 = 真正过 HBM 的字节 / 耗时；片上那次读由 L2/shared 承担，不计入。"
          "占上界 = 理论上界 / 实测。）")

    # ---- Roofline 验收 ----
    print(f"\n【Roofline 验收】按 lab08 实测带宽 {MEASURED_BW:.0f} GB/s（= 第 8 篇的 β）")
    print(f"  理论上界（1 读 + 1 写）      = {ideal_ms:.3f} ms")
    if fit:
        got = timing["softmax_shmem_kernel"]
        assert got < base, "版本 3 不该比三段式慢"

    best = max(roofline_frac.values())
    if roofline_ok:
        assert best > 0.90, f"最好的实现也只达到理论上界的 {best * 100:.0f}%，kernel 有问题"
        assert best < 1.10, f"达到上界的 {best * 100:.0f}%，超过 100% 说明口径错了（数据被缓存？）"
        print(f"  最好的实现达到理论上界的 {best * 100:.0f}% → 确实跑在 Roofline 上")
        print(f"  三段式 {base / ideal_ms:.2f}x 于上界 —— 这就是 FlashAttention 要消灭的浪费")
    else:
        print(f"  （尺寸太小，跳过 Roofline 断言）")

    # ---- 占用率诊断 ----
    if roofline_ok and fit and roofline_frac["softmax_shmem_kernel"] < 0.90:
        print(f"\n【为什么 shmem 版掉下来了】")
        print(f"  行缓存要 {need_shared / 1024:.1f} KB shared/block，"
              f"每个 SM 只能常驻 {occ['softmax_shmem_kernel']} 个 block"
              f"（fused 版只要 {2 * BLOCK * 4 / 1024:.1f} KB，可常驻 {occ['softmax_fused_kernel']} 个）。")
        print(f"  占用率不够就掩盖不住访存延迟 → 只能跑出 "
              f"{roofline_frac['softmax_shmem_kernel'] * 100:.0f}% 的上界。")
        print(f"  这正是 FlashAttention 不对整行做缓存、而是【分块】的原因：")
        print(f"  块小到能常驻 → 占用率不掉 → 才能真正贴着带宽跑。")
        print(f"  本例里 N <= 4096 时行放得下且占用率够，shmem 版能到 98%；N 一大就崩。")

    # ---- 对照 PyTorch ----
    try:
        import torch
        t = torch.from_numpy(x).cuda()

        ms_t = timeit(lambda: torch.softmax(t, dim=-1))
        bw_t = tensor_bytes * 2 / (ms_t * 1e-3) / 1e9
        print(f"\n【对照】torch.softmax(dim=-1) = {ms_t:.3f} ms"
              f"（按 1 读 + 1 写算，有效带宽 {bw_t:.1f} GB/s）")
        if fit:
            print(f"        softmax_shmem / torch  = {timing['softmax_shmem_kernel'] / ms_t:.2f}x")
    except ImportError:
        pass

    for b in (d_x, d_y, d_m, d_s):
        b.free()
    driver.cuModuleUnload(mod)
    print("\n完成。源码：lab09_flash_attention/cuda_softmax/softmax_kernels.cu")


if __name__ == "__main__":
    main()
