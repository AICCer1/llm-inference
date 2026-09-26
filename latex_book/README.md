# 《大模型推理系统与核心原理》全栈专著 LaTeX PDF

本目录包含将全套《大模型推理系统与核心原理》教程（理论推导篇 00~13、工业实战源码篇 Lab 03~13 及 Capstone）编译为出版级专著 PDF 的完整 LaTeX 源码与自动化构建脚本。

## 📖 核心成果与直接获取

* **最终生成专著 PDF**：[`main.pdf`](main.pdf)
  * **页数**：264 页
  * **结构**：全双栏详细目录、全书签超链接、数学定理彩色环境、TikZ 原生矢量架构图
  * **覆盖**：14 篇深度数学推导 + 14 组工业级实操源码精解

## 🛠️ 目录结构说明

```
latex_book/
├── main.pdf              # 最终编译完成的高清矢量 PDF 专著 (264 页)
├── main.tex              # LaTeX 专著主控入口文件
├── preamble.tex          # 宏包配置、版面几何、XeLaTeX 中西文字体、代码与彩色高亮环境
├── build_book.py         # 专著自动化构建脚本 (将 Markdown、数学公式、源码与 TikZ 图自动转换为 TeX)
├── compile.sh            # 自动化双遍 XeLaTeX 编译脚本
├── chapters/             # 第一部分：理论推导篇 TeX 文件 (前言 + ch00~ch13)
├── labs/                 # 第二部分：工业实战与源码精解 TeX 文件 (lab00~lab13)
└── code/                 # 实验配套高亮 Python 核心源码库
```

## 🚀 本地重新构建指南

### 1. 环境依赖（以 Ubuntu / Debian 为例）

```bash
sudo apt update
sudo apt install -y texlive-xetex texlive-lang-chinese texlive-latex-extra texlive-fonts-recommended fonts-noto-cjk
```

### 2. 生成 TeX 章节源码与主控文件

```bash
python3 build_book.py
```

### 3. 双遍编译生成 PDF

```bash
chmod +x compile.sh
./compile.sh
```
编译成功后，将在当前目录下生成最新的 `main.pdf`。
