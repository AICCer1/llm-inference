"""
Lab 05 - 训练: 用本仓库自己的教程文本，训练一个字符级的迷你 Llama

对应教程: 第 3 篇（交叉熵、困惑度、Adam、初始损失检查）、第 5 篇（架构）

运行（GPU 约 0.5~1.5 分钟，取决于显卡时钟状态；CPU 可加 --device cpu --steps 300，效果差一些但能跑通）:
  .venv/bin/python lab05_train_tiny_llama/train.py
产物:
  lab05_train_tiny_llama/tiny_llama.pt   (权重 + 配置 + 字表，供 generate.py 使用)
"""

import argparse
import glob
import math
import os
import time

import torch
import torch.nn.functional as F

from model import Config, TinyLlama

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_corpus():
    files = sorted(glob.glob(os.path.join(ROOT, "zero_to_hero_tutorial", "*.md")))
    files += sorted(glob.glob(os.path.join(ROOT, "*", "README.md")))
    files.append(os.path.join(ROOT, "README.md"))
    text = "\n\n".join(open(f, encoding="utf-8").read() for f in files if os.path.exists(f))
    return text, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=128, help="训练时的上下文长度")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2, help="数据太少，用 dropout 缓解过拟合")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device

    # ------------------------------------------------------------------
    # 1. 数据：字符级分词（最简单的 tokenizer：每个字符一个 id）
    # ------------------------------------------------------------------
    text, n_files = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    # 按 1000 字符一段切开，每 10 段取 1 段做验证集：让验证集覆盖所有章节，而不是只取文末的几个 README
    chunks = list(torch.split(data, 1000))
    train_data = torch.cat([c for i, c in enumerate(chunks) if i % 10 != 5])
    val_data = torch.cat([c for i, c in enumerate(chunks) if i % 10 == 5])
    V = len(chars)
    print(f"📚 语料: {n_files} 个 Markdown 文件，共 {len(text):,} 个字符，字表大小 V = {V}")
    print(f"   训练集 {len(train_data):,} 字符 / 验证集 {len(val_data):,} 字符")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block - 1, (args.batch,))
        x = torch.stack([d[i:i + args.block] for i in ix])
        y = torch.stack([d[i + 1:i + args.block + 1] for i in ix])   # 目标 = 输入右移一位（预测下一个字符）
        return x.to(dev), y.to(dev)

    # ------------------------------------------------------------------
    # 2. 模型
    # ------------------------------------------------------------------
    cfg = Config(vocab_size=V, dropout=args.dropout)
    model = TinyLlama(cfg).to(dev)
    print(f"🧠 模型: {cfg.n_layer} 层, d_model={cfg.d_model}, Q 头={cfg.n_head}, KV 头={cfg.n_kv_head} (GQA), "
          f"FFN={cfg.ffn_hidden}, dropout={cfg.dropout}, 参数量 {model.num_params() / 1e6:.2f}M")

    use_bf16 = dev.startswith("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def estimate_loss(iters=20):
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(iters):
                x, y = get_batch(split)
                with autocast:
                    logits = model(x)
                losses.append(F.cross_entropy(logits.float().view(-1, V), y.view(-1)).item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # ------------------------------------------------------------------
    # 3. 训练前检查：初始损失应 ≈ ln(V)（第 3 篇 §5.3 的调试黄金法则）
    # ------------------------------------------------------------------
    init = estimate_loss(5)
    print(f"\n🔍 初始损失检查: 实测 {init['val']:.3f} vs 理论 ln(V) = {math.log(V):.3f}  "
          f"({'✅ 正常' if abs(init['val'] - math.log(V)) < 0.5 else '❌ 偏差过大，检查初始化'})")
    assert abs(init['val'] - math.log(V)) < 0.5, '初始损失应 ≈ ln(V)，检查初始化'

    # ------------------------------------------------------------------
    # 4. 训练循环：AdamW + 线性预热 + 余弦退火
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(1, args.steps // 30)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    print(f"\n🏋️ 开始训练 {args.steps} 步 (batch={args.batch}, block={args.block}, device={dev})")
    path = os.path.join(HERE, "tiny_llama.pt")
    best = {"val": float("inf"), "step": -1, "train": None}
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            l = estimate_loss()
            mark = ""
            if l["val"] < best["val"]:
                # 只保存验证集损失最低的那一刻（early stopping），之后的"进步"只是在背训练集
                best = {"val": l["val"], "step": step, "train": l["train"]}
                torch.save({"model": model.state_dict(), "config": cfg.__dict__, "chars": chars}, path)
                mark = "  💾"
            print(f"   step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} | "
                  f"val PPL {math.exp(l['val']):7.2f} | {time.time() - t0:5.1f}s{mark}")
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch("train")
        with autocast:
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)      # 梯度是累加的，每步清零（Lab 03 实验 6）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\n💡 解读:")
    print(f"   1. 验证集 PPL 从约 {V}（均匀瞎猜）降到了 {math.exp(best['val']):.1f}：模型确实学到了中文字符与 Markdown 的统计规律。")
    print(f"   2. 训练损失一路下降，验证损失却在某一步后不降反升 —— 这就是【过拟合】：")
    print(f"      {model.num_params() / 1e6:.1f}M 参数 vs 只有 {len(train_data) / 1e3:.0f}K 个训练字符，模型容量远大于数据量，开始逐字背诵训练集。")
    print(f"      第 5 篇讲的 Chinchilla 比例（约 20 token/参数）在这里只有 {len(train_data) / model.num_params():.3f}，差了几百倍。")
    print(f"   3. 所以我们只保存了验证损失最低的第 {best['step']} 步（early stopping）。")
    print(f"\n💾 已保存到 {os.path.relpath(path, ROOT)}，下一步运行: .venv/bin/python lab05_train_tiny_llama/generate.py")


if __name__ == "__main__":
    main()
