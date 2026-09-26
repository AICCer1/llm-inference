"""
毕业设计: 对任何 OpenAI 兼容的推理服务（vLLM / SGLang / llama.cpp 的 llama-server / TGI）做延迟与吞吐压测 (零外部依赖)

对应教程: zero_to_hero_tutorial/12_推理服务系统：PagedAttention、前缀缓存与调度.md（§1 指标体系）

测量内容（第 12 篇 §1）:
  - TTFT：从发出请求到收到第一个 token
  - TPOT：相邻两个输出 token 的平均间隔（每个请求算一个值）
  - 端到端延迟、系统总输出吞吐（token/s）
  - 在不同并发数下重复，观察"吞吐-延迟权衡"（第 8 篇 §4.4）

[注意] 测量前先读这三条，否则数字会骗你（每一条都会让结果系统性偏乐观）：
  1. **前缀缓存**：所有并发档位复用同一批问题，档位之间不清缓存；vLLM 默认开启前缀缓存，
     所以靠后的档位 TTFT 会偏低。要干净的数字，请在各档位之间重启服务，或换用互不相同的问题。
  2. **输出长度**：默认带 `ignore_eos: true`，让每个请求都恰好产出 `--max-tokens` 个 token，
     否则"总吞吐 tok/s"实际测的是"模型有多话唠"而不是引擎有多快。
     注意这个字段是 vLLM/SGLang 的扩展，标准 OpenAI 服务可能忽略它——那就只能靠 max_tokens 一致来近似。
  3. **预热**：脚本会先发一轮不计入统计的预热请求。GPU 从空闲频率爬到稳态需要时间，
     不预热的第一档会明显偏慢（见 lab08 README 里 217~380 GB/s 的实测差异）。

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
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(
            f"[X] 连不上 {base_url}/v1/models：{e}\n"
            f"   先启动一个 OpenAI 兼容的推理服务，例如：\n"
            f"     vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096\n"
            f"     llama-server -m <model>.gguf -ngl 99 -c 8192 --port 8000\n"
            f"   再重新运行本脚本（默认端口 8000）。") from None


def one_request(base_url, model, prompt, max_tokens, system, ignore_eos=True):
    """发送一个流式请求，返回 (ttft, 各 token 到达时间列表, 输出 token 数)"""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        # vLLM / SGLang 的扩展：忽略 EOS，保证每个请求输出长度一致，吞吐才可比
        payload["ignore_eos"] = True
    body = json.dumps(payload).encode()
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
    # 预热：不计入统计。GPU 从空闲频率爬到稳态要时间，不预热的第一档会明显偏慢。
    for _ in range(args.warmup):
        try:
            one_request(args.base_url, model, QUESTIONS[0], min(args.max_tokens, 32), system,
                        ignore_eos=args.ignore_eos)
        except (urllib.error.URLError, OSError) as e:
            print(f"   并发 {concurrency}: 预热失败 {e}")
            return

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
                r = one_request(args.base_url, model, QUESTIONS[i % len(QUESTIONS)], args.max_tokens, system,
                                ignore_eos=args.ignore_eos)
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
    ttft = [r[0] for r in ok if r[0] == r[0]]          # 过滤 NaN（没收到任何 content 分片的请求）
    # TPOT 的分子是「内容分片到达时间的跨度」，跨过的是 len(r[1])-1 个间隔；
    # 分母必须用同一个数。用 usage.completion_tokens 会和服务端是否一个 chunk 带多个 token 耦合，
    # 那样算出来的 TPOT 口径不一致（llama.cpp 等会一个 chunk 塞多个 token，TPOT 会被系统性高估）。
    tpot, n_mismatch = [], 0
    for r in ok:
        if len(r[1]) > 1:
            tpot.append((r[1][-1] - r[1][0]) / (len(r[1]) - 1))
            if r[2] != len(r[1]):
                n_mismatch += 1
    out_tokens = sum(r[2] for r in ok)

    if not ttft or not tpot:
        print(f"   {concurrency:>4} | {len(ok):>4} | 样本不足（TTFT {len(ttft)} 个 / TPOT {len(tpot)} 个），"
              f"调大 --num-requests 或 --max-tokens")
        return
    if n_mismatch:
        print(f"   [注意] 并发 {concurrency}: {n_mismatch}/{len(ok)} 个请求的 usage.completion_tokens "
              f"与收到的内容分片数不一致（一个 chunk 带了多个 token？）—— TPOT 已按分片间隔计算。")
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
    ap.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                    help="让每个请求都恰好输出 --max-tokens 个 token，吞吐才可比（vLLM/SGLang 支持；"
                         "标准 OpenAI 服务会忽略该字段）。用 --no-ignore-eos 关闭。")
    ap.add_argument("--warmup", type=int, default=2,
                    help="每个并发档位开测前先发这么多个不计入统计的请求，把 GPU 频率拉起来")
    ap.add_argument("--shared-prefix-tokens", type=int, default=0,
                    help=">0 时所有请求共享一段约这么多 token 的系统提示词，用来观察前缀缓存对 TTFT 的影响")
    args = ap.parse_args()

    model = args.model or get_model(args.base_url)
    system = None
    if args.shared_prefix_tokens > 0:
        # 大约每个汉字 1 个 token；内容固定不变，才能命中前缀缓存（第 12 篇 §5）
        system = ("你是一个严谨的大模型推理系统专家。" * (args.shared_prefix_tokens // 16 + 1))[: args.shared_prefix_tokens]
    print(f"[目标] 服务 {args.base_url} | 模型 {model} | 每档 {args.num_requests} 个请求 | max_tokens={args.max_tokens}"
          + (f" | 共享前缀约 {args.shared_prefix_tokens} token" if system else ""))
    print(f"   {'并发':>4} | {'完成':>4} | {'TTFT P50':>8} | {'TTFT P99':>8} | {'TPOT P50':>8} | {'TPOT P99':>8} | "
          f"{'总吞吐 tok/s':>10} | {'单流 tok/s':>8}")
    print(f"   {'':>4} | {'':>4} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | {'':>10} | {'':>8}")
    for c in args.concurrency:
        run_level(args, model, c, system)
    print("\n[思考] 对照第 8 篇：并发 1 时的单流速度应接近（但低于）带宽下界 = 实测带宽 / 权重字节数；")
    print("   并发增大时总吞吐上升、单流速度与 TTFT 变差 —— 这就是吞吐-延迟权衡。")


if __name__ == "__main__":
    main()
