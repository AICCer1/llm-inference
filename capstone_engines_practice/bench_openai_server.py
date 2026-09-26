"""
毕业设计: 对任何 OpenAI 兼容的推理服务（vLLM / SGLang / llama.cpp 的 llama-server / TGI）做延迟与吞吐压测 (零外部依赖)

对应教程: zero_to_hero_tutorial/10_推理服务系统（§1 指标体系）

测量内容（第 12 篇 §1）:
  - TTFT：从发出请求到收到第一个 token
  - TPOT：相邻两个输出 token 的平均间隔（每个请求算一个值）
  - 端到端延迟、系统总输出吞吐（token/s）
  - 在不同并发数下重复，观察"吞吐-延迟权衡"（第 8 篇 §4.4）

用法示例:
  # 1) 先启动服务，例如
  #    vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096
  #    llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf -ngl 99 -c 8192 --port 8000
  # 2) 再压测
  python3 capstone_engines_practice/bench_openai_server.py --base-url http://localhost:8000 --concurrency 1 4 16 32
  # 测前缀缓存：所有请求共享一段很长的系统提示词
  python3 capstone_engines_practice/bench_openai_server.py --shared-prefix-tokens 2000 --concurrency 8
"""

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request

QUESTIONS = [
    "用三句话解释什么是 KV Cache。",
    "为什么大模型的 Decode 阶段是访存受限的？",
    "简要介绍 PagedAttention 解决了什么问题。",
    "投机解码为什么不会改变输出分布？",
    "比较 GPTQ 和 AWQ 两种量化方法。",
    "什么是 RoPE 旋转位置编码？",
    "解释连续批处理与静态批处理的区别。",
    "FlashAttention 为什么能减少显存占用？",
]


def get_model(base_url):
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def one_request(base_url, model, prompt, max_tokens, system):
    """发送一个流式请求，返回 (ttft, 各 token 到达时间列表, 输出 token 数)"""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    arrivals, usage_tokens = [], None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens")
            for ch in chunk.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    arrivals.append(time.perf_counter() - t0)
    n_tokens = usage_tokens or len(arrivals)   # 有 usage 就用精确 token 数；否则按流式分片数近似
    return (arrivals[0] if arrivals else float("nan")), arrivals, n_tokens, time.perf_counter() - t0


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def run_level(args, model, concurrency, system):
    results, lock = [], threading.Lock()
    n_total = max(args.num_requests, concurrency)
    counter = iter(range(n_total))

    def worker():
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            try:
                r = one_request(args.base_url, model, QUESTIONS[i % len(QUESTIONS)], args.max_tokens, system)
            except (urllib.error.URLError, OSError) as e:
                r = e
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if not isinstance(r, Exception)]
    if not ok:
        print(f"   并发 {concurrency}: 全部失败，例如 {results[0]}")
        return
    ttft = [r[0] for r in ok]
    tpot = [(r[1][-1] - r[1][0]) / (r[2] - 1) for r in ok if r[2] > 1 and len(r[1]) > 1]
    out_tokens = sum(r[2] for r in ok)
    print(f"   {concurrency:>4} | {len(ok):>4} | {statistics.median(ttft) * 1e3:8.0f} | {pct(ttft, 99) * 1e3:8.0f} | "
          f"{statistics.median(tpot) * 1e3:8.1f} | {pct(tpot, 99) * 1e3:8.1f} | {out_tokens / wall:10.1f} | "
          f"{1 / statistics.median(tpot):8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="默认取 /v1/models 返回的第一个")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--num-requests", type=int, default=32, help="每个并发档位总共发多少个请求")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--shared-prefix-tokens", type=int, default=0,
                    help=">0 时所有请求共享一段约这么多 token 的系统提示词，用来观察前缀缓存对 TTFT 的影响")
    args = ap.parse_args()

    model = args.model or get_model(args.base_url)
    system = None
    if args.shared_prefix_tokens > 0:
        # 大约每个汉字 1 个 token；内容固定不变，才能命中前缀缓存（第 12 篇 §5）
        system = ("你是一个严谨的大模型推理系统专家。" * (args.shared_prefix_tokens // 16 + 1))[: args.shared_prefix_tokens]
    print(f"🎯 服务 {args.base_url} | 模型 {model} | 每档 {args.num_requests} 个请求 | max_tokens={args.max_tokens}"
          + (f" | 共享前缀约 {args.shared_prefix_tokens} token" if system else ""))
    print(f"   {'并发':>4} | {'完成':>4} | {'TTFT P50':>8} | {'TTFT P99':>8} | {'TPOT P50':>8} | {'TPOT P99':>8} | "
          f"{'总吞吐 tok/s':>10} | {'单流 tok/s':>8}")
    print(f"   {'':>4} | {'':>4} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | {'':>10} | {'':>8}")
    for c in args.concurrency:
        run_level(args, model, c, system)
    print("\n💡 对照第 8 篇：并发 1 时的单流速度应接近（但低于）带宽下界 = 实测带宽 / 权重字节数；")
    print("   并发增大时总吞吐上升、单流速度与 TTFT 变差 —— 这就是吞吐-延迟权衡。")


if __name__ == "__main__":
    main()
