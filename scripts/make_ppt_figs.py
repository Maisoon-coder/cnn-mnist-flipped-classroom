# -*- coding: utf-8 -*-
"""生成第 8—10 页 PPT 素材（组员 4）。

产出（写入 figures/）：
    fig10_network.png        网络结构图（含逐层形状与参数量）
    fig11_training_loop.png  训练循环五步与代码对应
    fig12_predict_demo.png   单图预测：输入图 + 10 类概率

用法（项目根目录，需装 torch / matplotlib）：
    python scripts/make_ppt_figs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 与组员 5 的图统一配色
FC_COLOR = "#3B6FB6"
CNN_COLOR = "#D9662B"
GRAY = "#5F6368"


# ---------------------------------------------------------------- 图 10：网络结构
def fig_network():
    stages = [
        ("输入", "1 × 28 × 28", "灰度图", "#EDF2F7", "#4A5568"),
        ("Conv1", "6 × 28 × 28", "5×5 核, padding=2\n156 参数", "#FDE8DC", CNN_COLOR),
        ("MaxPool", "6 × 14 × 14", "2×2, 步长 2\n0 参数", "#EDF2F7", "#4A5568"),
        ("Conv2", "16 × 10 × 10", "5×5 核, padding=0\n2,416 参数", "#FDE8DC", CNN_COLOR),
        ("MaxPool", "16 × 5 × 5", "2×2, 步长 2\n0 参数", "#EDF2F7", "#4A5568"),
        ("Flatten + 全连接", "400 → 120 → 84", "48,120 + 10,164\n参数", "#FBE4D5", CNN_COLOR),
        ("输出", "10", "logits（不加 Softmax）\n850 参数", "#EDF2F7", "#4A5568"),
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(15.5, 4.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    bw, bh = 0.115, 0.38
    ys = 0.58
    xs = [0.075 + i * 0.1415 for i in range(n)]

    for (title, shape, note, fc, ec), x in zip(stages, xs):
        ax.add_patch(FancyBboxPatch((x - bw / 2, ys - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    facecolor=fc, edgecolor=ec, linewidth=2, zorder=3))
        ax.text(x, ys + 0.125, title, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=ec, zorder=4)
        ax.text(x, ys - 0.005, shape, ha="center", va="center", fontsize=11, zorder=4)
        ax.text(x, ys - 0.115, note, ha="center", va="center", fontsize=8.2,
                color=GRAY, linespacing=1.5, zorder=4)

    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + bw / 2 + 0.004, ys),
                                     (xs[i + 1] - bw / 2 - 0.004, ys),
                                     arrowstyle="-|>", mutation_scale=15,
                                     color=GRAY, lw=1.8, zorder=2))

    ax.text(0.5, 0.955, "简易 CNN 的结构：一次前向做了什么", ha="center",
            fontsize=17, fontweight="bold")
    ax.text(0.5, 0.20, "合计 61,706 个参数　｜　对照的全连接基线：203,530 个参数（约 3.3 倍）",
            ha="center", fontsize=11.5, color=CNN_COLOR, fontweight="bold")
    ax.text(0.5, 0.10,
            "池化层没有可学习参数；卷积后都接了 ReLU（图中省略）；尺寸按 floor((H+2P−K)/S)+1 计算",
            ha="center", fontsize=9, color=GRAY)

    out = FIG_DIR / "fig10_network.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out.relative_to(ROOT))


# ---------------------------------------------------------------- 图 11：训练循环
def fig_training_loop():
    steps = [
        ("① 清空旧梯度", "optimizer.zero_grad()",
         "PyTorch 的梯度是累加的；\n不清零会叠加上一批的梯度"),
        ("② 前向传播", "logits = model(x)",
         "一批 64 张图 → 64×10 的原始分数"),
        ("③ 计算损失", "loss = loss_fn(logits, y)",
         "交叉熵内部已含 log_softmax，\n模型输出不要再套 Softmax"),
        ("④ 反向传播", "loss.backward()",
         "只求梯度，还不改参数"),
        ("⑤ 更新参数", "optimizer.step()",
         "w ← w − lr × grad，\n这一步才真正修改权重"),
    ]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(15.5, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    bw, bh = 0.155, 0.44
    ys = 0.56
    xs = [0.10 + i * 0.20 for i in range(n)]

    for (title, code, note), x in zip(steps, xs):
        ax.add_patch(FancyBboxPatch((x - bw / 2, ys - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.008,rounding_size=0.025",
                                    facecolor="#FDE8DC", edgecolor=CNN_COLOR,
                                    linewidth=2, zorder=3))
        ax.text(x, ys + 0.155, title, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=CNN_COLOR, zorder=4)
        ax.text(x, ys + 0.045, code, ha="center", va="center", fontsize=9.2,
                family="monospace", zorder=4,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#E2C9B8", lw=0.8))
        ax.text(x, ys - 0.115, note, ha="center", va="center", fontsize=8.0,
                color=GRAY, linespacing=1.5, zorder=4)

    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + bw / 2 + 0.004, ys),
                                     (xs[i + 1] - bw / 2 - 0.004, ys),
                                     arrowstyle="-|>", mutation_scale=15,
                                     color=GRAY, lw=1.8, zorder=2))

    # 回到第一步的循环箭头（走方框下方，避免穿过图形）
    loop_y = 0.215
    bottom = ys - bh / 2 - 0.012
    ax.plot([xs[-1], xs[-1]], [bottom, loop_y], color="#1A73E8", lw=2.0,
            linestyle=(0, (5, 3)), zorder=2)
    ax.plot([xs[-1], xs[0]], [loop_y, loop_y], color="#1A73E8", lw=2.0,
            linestyle=(0, (5, 3)), zorder=2)
    ax.annotate("", xy=(xs[0], bottom), xytext=(xs[0], loop_y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=16, color="#1A73E8",
                                lw=2.0, ls=(0, (5, 3)), shrinkA=0, shrinkB=0), zorder=2)
    ax.text(0.5, 0.115, "下一批（共 938 批 = 1 个 epoch，60000 ÷ 64；末尾不足 64 张的一批也保留）",
            ha="center", fontsize=10, color="#1A73E8")

    ax.text(0.5, 0.955, "一次训练更新发生了什么（对应 src/train.py 的循环体）",
            ha="center", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.04, "轮末再用 model.eval() + torch.no_grad() 评估训练集与测试集，不更新参数",
            ha="center", fontsize=9, color=GRAY)

    out = FIG_DIR / "fig11_training_loop.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out.relative_to(ROOT))


# ---------------------------------------------------------------- 图 12：单图预测
def fig_predict():
    import torch
    from PIL import Image

    from src import data as D
    from src import models as M

    ckpt = ROOT / "outputs/main_seed42_v1/checkpoints/cnn_seed42_last.pt"
    img_path = ROOT / "demo/digit7_test0.png"
    model, meta = M.load_model_from_checkpoint(ckpt)
    tf = D.build_transform()

    img = Image.open(img_path).convert("L")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).tolist()
    pred = int(max(range(10), key=lambda i: probs[i]))

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.05], wspace=0.22,
                          left=0.03, right=0.97, top=0.84, bottom=0.17)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img, cmap="gray", interpolation="nearest")
    ax0.set_xticks([]); ax0.set_yticks([])
    for s in ax0.spines.values():
        s.set_edgecolor("#4A5568"); s.set_linewidth(2)
    ax0.set_title("输入：28×28 灰度图\n（MNIST 测试集第 0 张）", fontsize=11, color="#202124")

    ax1 = fig.add_subplot(gs[1])
    colors = ["#D9662B" if i == pred else "#BFC9D4" for i in range(10)]
    bars = ax1.barh(range(10), [p * 100 for p in probs], color=colors, height=0.72)
    ax1.set_yticks(range(10))
    ax1.set_yticklabels([str(i) for i in range(10)], fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("模型给出的概率（%）", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)
    ax1.set_axisbelow(True)
    for b, p in zip(bars, probs):
        ax1.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2,
                 "{:.2f}%".format(p * 100), va="center", fontsize=9, color="#202124")

    fig.suptitle("单图预测：读权重 → 同一套预处理 → 前向 → 取最大概率",
                 fontsize=16, fontweight="bold", y=0.96)
    fig.text(0.5, 0.04,
             "模型：cnn_seed42_last.pt（seed=42, epoch=5, 测试准确率 99.00%）　｜　"
             "预测结果：{}　｜　最高概率 {:.2f}%".format(pred, max(probs) * 100),
             ha="center", fontsize=9.5, color=GRAY)

    out = FIG_DIR / "fig12_predict_demo.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    print("saved:", out.relative_to(ROOT), "| pred =", pred,
          "| top prob = {:.4f}".format(max(probs)))


if __name__ == "__main__":
    fig_network()
    fig_training_loop()
    fig_predict()
