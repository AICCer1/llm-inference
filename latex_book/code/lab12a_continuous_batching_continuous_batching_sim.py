"""
Lab 12a: 静态 Batching vs 连续批处理 (Continuous Batching) 离散事件仿真器 (零外部依赖)

本脚本直观演示：
1. 静态批处理 (Static Batching): 必须等待 Batch 内所有请求全部生成结束，长尾请求导致严重的 Padding 气泡与排队延迟。
2. 连续批处理 (Continuous Batching / In-flight Batching): 迭代级调度，短请求一旦结束立即释放槽位，等待中的新请求立刻补位。
3. 统计两者的系统吞吐量 (Throughput)、首字排队耗时 (TTFT 排队延迟) 与算力槽位浪费率 (Bubble Ratio)。
"""

import collections


class Request:
    def __init__(self, req_id, arrival_time, output_len):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.output_len = output_len
        self.tokens_generated = 0
        self.start_time = None
        self.finish_time = None

    def is_finished(self):
        return self.tokens_generated >= self.output_len


def simulate_static_batching(requests_data, max_batch_size=4):
    """
    静态批处理模式:
    每次必须凑足（或取满）max_batch_size 个请求作为一个固定批次，
    批次内所有请求必须等待最长请求完成后，才能释放槽位迎接下一批。
    """
    queue = [Request(r["id"], r["arrival"], r["output_len"]) for r in requests_data]
    current_time = 0
    completed = []
    
    total_effective_tokens = sum(r.output_len for r in queue)
    total_slot_steps = 0 # 统计占用的槽位步数

    while queue:
        # 挑选已到达的请求
        ready = [r for r in queue if r.arrival_time <= current_time]
        if not ready:
            # 快进时间到下一个到达的请求
            current_time = queue[0].arrival_time
            ready = [queue[0]]
            
        batch = ready[:max_batch_size]
        for r in batch:
            queue.remove(r)
            r.start_time = current_time

        # 批次中最长生成长度
        max_len = max(r.output_len for r in batch)
        
        # 静态执行 max_len 步
        total_slot_steps += max_len * len(batch)
        current_time += max_len
        
        for r in batch:
            r.tokens_generated = r.output_len
            r.finish_time = current_time
            completed.append(r)

    makespan = current_time
    # 算力浪费率: 实际生成的 token 步数 vs 实际占用的槽位步数
    bubble_ratio = (total_slot_steps - total_effective_tokens) / total_slot_steps if total_slot_steps > 0 else 0
    return completed, makespan, bubble_ratio


def simulate_continuous_batching(requests_data, max_batch_size=4):
    """
    连续批处理模式 (Continuous Batching / In-flight Batching):
    每一步迭代 (Iteration) 检查：
    - 是否有请求生成完毕？若有，立即退场！
    - 是否有空余槽位且等待队列中有新请求已到达？若有，立即加入当前步！
    """
    pending = collections.deque([Request(r["id"], r["arrival"], r["output_len"]) for r in requests_data])
    active_slots = []
    completed = []
    current_time = 0
    
    total_effective_tokens = sum(r["output_len"] for r in requests_data)
    total_slot_steps = 0

    while pending or active_slots:
        # 1. 尝试将已到达的新请求补充进空闲槽位
        while len(active_slots) < max_batch_size and pending:
            if pending[0].arrival_time <= current_time:
                req = pending.popleft()
                if req.start_time is None:
                    req.start_time = current_time
                active_slots.append(req)
            else:
                break

        # 如果槽位全空且没有就绪请求，快进时间
        if not active_slots:
            current_time = pending[0].arrival_time
            continue

        # 2. 执行单步生成 (Iteration Step)
        total_slot_steps += len(active_slots)
        current_time += 1
        
        # 3. 检查每个槽位的生成进度
        survivors = []
        for req in active_slots:
            req.tokens_generated += 1
            if req.is_finished():
                req.finish_time = current_time
                completed.append(req)
            else:
                survivors.append(req)
                
        active_slots = survivors

    makespan = current_time
    bubble_ratio = (total_slot_steps - total_effective_tokens) / total_slot_steps if total_slot_steps > 0 else 0
    return completed, makespan, bubble_ratio


def run_simulation():
    print("=" * 72)
    print("[起步] 静态批处理 (Static) vs 连续批处理 (Continuous) 调度性能仿真")
    print("=" * 72)
    
    # 模拟 10 个具有不同到达时间与输出长度的并发请求
    # 包含典型长尾负载：部分请求 5-15 个 token 极快结束，部分需要 80 个 token
    requests_workload = [
        {"id": 1,  "arrival": 0,  "output_len": 15},
        {"id": 2,  "arrival": 0,  "output_len": 70}, # 长请求
        {"id": 3,  "arrival": 0,  "output_len": 10},
        {"id": 4,  "arrival": 0,  "output_len": 25},
        {"id": 5,  "arrival": 10, "output_len": 80}, # 稍后到达的长请求
        {"id": 6,  "arrival": 12, "output_len": 8},
        {"id": 7,  "arrival": 20, "output_len": 18},
        {"id": 8,  "arrival": 25, "output_len": 12},
        {"id": 9,  "arrival": 30, "output_len": 65},
        {"id": 10, "arrival": 35, "output_len": 20},
    ]

    total_tokens = sum(r["output_len"] for r in requests_workload)
    max_batch = 4
    
    print(f"[重点] 测试负载: {len(requests_workload)} 个请求 | 总 Token 需求: {total_tokens} | 槽位容量 (Max Batch): {max_batch}")
    print(f"[重点] 请求分布:")
    for r in requests_workload:
        print(f"   - 请求 #{r['id']:2d}: 到达时间 t={r['arrival']:2d} | 目标输出长度 = {r['output_len']:2d} tokens")
    print("-" * 72)

    # 1. 静态批处理仿真
    c_static, makespan_static, bubble_static = simulate_static_batching(requests_workload, max_batch_size=max_batch)
    avg_wait_static = sum(r.start_time - r.arrival_time for r in c_static) / len(c_static)
    avg_turnaround_static = sum(r.finish_time - r.arrival_time for r in c_static) / len(c_static)
    throughput_static = total_tokens / makespan_static

    # 2. 连续批处理仿真
    c_cont, makespan_cont, bubble_cont = simulate_continuous_batching(requests_workload, max_batch_size=max_batch)
    avg_wait_cont = sum(r.start_time - r.arrival_time for r in c_cont) / len(c_cont)
    avg_turnaround_cont = sum(r.finish_time - r.arrival_time for r in c_cont) / len(c_cont)
    throughput_cont = total_tokens / makespan_cont

    # 3. 对比总结
    print("\n[图表] 调度算法核心指标对比:")
    print(f"   {'指标项':<24} | {'静态批处理 (Static)':<22} | {'连续批处理 (Continuous)':<22}")
    print(f"   {'-'*24}-+-{'-'*22}-+-{'-'*22}")
    print(f"   {'总处理耗时 (Makespan)':<24} | {makespan_static:<22} 步 | {makespan_cont:<22} 步")
    print(f"   {'系统总吞吐量 (Throughput)':<24} | {throughput_static:<20.2f} tok/步 | {throughput_cont:<20.2f} tok/步")
    print(f"   {'平均首字排队等待时间':<24} | {avg_wait_static:<22.2f} 步 | {avg_wait_cont:<22.2f} 步")
    print(f"   {'平均端到端周转耗时':<24} | {avg_turnaround_static:<22.2f} 步 | {avg_turnaround_cont:<22.2f} 步")
    print(f"   {'Padding 无效计算占比':<24} | {bubble_static * 100:<21.1f}% | {bubble_cont * 100:<21.1f}%")
    util_static = total_tokens / (max_batch * makespan_static)
    util_cont = total_tokens / (max_batch * makespan_cont)
    print(f"   {'槽位占用率 (有效token/容量)':<22} | {util_static * 100:<21.1f}% | {util_cont * 100:<21.1f}%")
    # [注意] 注意 bubble_cont 恒等于 0.0 —— 它是【构造使然】，不是优化出来的结果：
    #    上面的 total_slot_steps += len(active_slots) 与每个活跃槽 +1 个 token 是同一件事，
    #    所以 (x - x) / x 永远是 0。拿它当断言等于什么都没断言（这里就不写了）。
    # 真正可能因调度器写错而失败的，是下面这两条：
    assert util_cont > util_static, f"连续批处理的槽位占用率 {util_cont:.1%} 应高于静态批处理的 {util_static:.1%}"
    assert makespan_cont < makespan_static, \
        f"连续批处理的总耗时 {makespan_cont} 应短于静态批处理的 {makespan_static}"

    print("\n[起步] 连续批处理核心收益:")
    print(f"   - 总耗时缩短 : {(1 - makespan_cont / makespan_static) * 100:.1f}%")
    print(f"   - 吞吐量提升 : {throughput_cont / throughput_static:.2f}x")
    print(f"   - 首字等待排队缩短 : {avg_wait_static / avg_wait_cont if avg_wait_cont > 0 else 999:.1f}x")

    print("\n[思考] 架构剖析:")
    print(f"   1. 静态批处理中，短请求做完后必须填充 Padding 陪跑：本负载下 {bubble_static * 100:.0f}% 的计算花在了 Padding 上。")
    print("   2. 连续批处理的 Padding 无效计算按构造就是 0（结束的请求当步就离开），所以这一行的对比只说明'它不陪跑'。")
    print(f"      更公平的对比是槽位占用率：{util_static * 100:.0f}% → {util_cont * 100:.0f}%。连续批处理也达不到 100%，")
    print("      剩下的空槽来自'没有新请求可补位'（请求到达得不够快）——此时空槽不算浪费，只是负载不满。")
    print("   3. 这正是为什么现代大模型推理服务（如 vLLM、TGI、TensorRT-LLM）必须将连续批处理作为默认底座！")
    print("=" * 72)


if __name__ == "__main__":
    run_simulation()
