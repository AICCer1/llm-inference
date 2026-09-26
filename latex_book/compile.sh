#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "开始编译《大模型推理系统与核心原理》LaTeX 专著 PDF..."
echo "=========================================================="

echo ">>> 正在执行第 1 遍 XeLaTeX 编译（构建交叉引用与结构）..."
xelatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

echo ">>> 正在执行第 2 遍 XeLaTeX 编译（完善目录与书签超链接）..."
xelatex -interaction=nonstopmode main.tex

if [ -f "main.pdf" ]; then
    echo "=========================================================="
    echo "🎉 恭喜！编译成功！"
    echo "生成的 PDF 路径: $DIR/main.pdf"
    ls -lh main.pdf
    echo "=========================================================="
else
    echo "❌ 编译失败，请检查 main.log"
    tail -n 40 main.log
    exit 1
fi
