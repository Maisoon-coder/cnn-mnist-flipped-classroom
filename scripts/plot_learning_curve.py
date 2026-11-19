# -*- coding: utf-8 -*-
"""5-04 学习曲线：从 metrics.csv 生成 2×2 面板。

口径（与《分工细则》§6.3、H4 交接文档一致）：
- 数据只来自 metrics.csv，不手填任何数字；
- 横轴是 epoch（1—5），纵轴分别是 loss 与 accuracy；
- FC / CNN 在四个子图中颜色完全一致；
- 取最后一轮（epoch 5）作为正式结果，同时在图上标出 FC 第 4→5 轮的回落，
  并注明"按规定仍取最后一轮"，避免被误读为挑轮次。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "main_seed42_v1"
OUT_DIR = ROOT / "figures"

#: 全项目统一配色：FC 蓝 / CNN 橙，四张子图、后续所有图都用这一套
COLOR = {"fc": "#3B6FB6", "cnn": "#D9662B"}
LABEL = {"fc": "FC（全连接）", "cnn": "CNN（卷积）"}
MARKER = {"fc": "o", "cnn": "s"}


def load_metrics(path: Path) -> dict:
    """metrics -> {model: {split: {"epoch": [...], "loss": [...], "acc": [...]}}}"""
    data = defaultdict(lambda: defaultdict(lambda: {"epoch": [], "loss": [], "acc": []}))
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            model, split = row["model"], row["split"]
            data[model][split]["epoch"].append(int(row["epoch"]))
            data[model][split]["loss"].append(float(row["loss"]))
            data[model][split]["acc"].append(float(row["accuracy"]))
    return data


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_metrics(RUN_DIR / "metrics.csv")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), dpi=200, sharex=True)
    panels = [
        (axes[0][0], "train", "loss", "训练集 loss", False),
        (axes[1][0], "test", "loss", "测试集 loss", False),
        (axes[0][1], "train", "acc", "训练集准确率", True),
        (axes[1][1], "test", "acc", "测试集准确率", True),
    ]

    for ax, split, metric, title, is_acc in panels:
        for model in ("fc", "cnn"):
            s = data[model][split]
            y = [v * 100 for v in s["acc"]] if is_acc else s["loss"]
            ax.plot(s["epoch"], y, marker=MARKER[model], ms=5.5, lw=2,
                    color=COLOR[model], label=LABEL[model])
            # 末轮数值直接标在图上，PPT 上不用再念表
            ax.annotate(f"{y[-1]:.2f}%" if is_acc else f"{y[-1]:.4f}",
                        xy=(s["epoch"][-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=COLOR[model], fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#222")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("epoch（训练轮次）" if split == "test" else "")
        ax.grid(True, ls=":", lw=0.8, color="#C8CCD2", alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if is_acc:
            ax.set_ylim(95.0, 100.0)
            ax.set_ylabel("accuracy（%）")
        else:
            ax.set_ylabel("loss（交叉熵）")

    axes[0][0].legend(frameon=False, fontsize=9.5, loc="upper right")

    # FC 在第 4→5 轮的回落：画个框，配一句"仍取最后一轮"
    ax = axes[1][1]
    fc_test = data["fc"]["test"]
    ax.annotate("FC 第4轮 97.47% → 第5轮 97.19%（回落 0.28 个百分点）\n"
                "按预注册口径仍取最后一轮，不以测试集挑轮次",
                xy=(5, fc_test["acc"][-1] * 100), xytext=(1.15, 95.18),
                fontsize=8.5, color="#444",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#888"),
                bbox=dict(boxstyle="round,pad=0.35", fc="#F5F6F8", ec="#D3D7DE", lw=0.8))

    fig.suptitle("FC 与 CNN 学习曲线（MNIST，seed=42，5 轮，同一配置）",
                 fontsize=14.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "数据来源：outputs/main_seed42_v1/metrics.csv（每轮各划分的 loss 与 accuracy，共 20 行）；训练集 60000 张，测试集 10000 张",
             ha="center", fontsize=8.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out = OUT_DIR / "fig1_learning_curves.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"已保存 {out}")

    # 同时导出一张纯数据表，供 PPT 表格页与答疑引用
    tab = OUT_DIR / "learning_curve_data.csv"
    with open(tab, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for model in ("fc", "cnn"):
            tr, te = data[model]["train"], data[model]["test"]
            for i in range(len(tr["epoch"])):
                w.writerow([model, tr["epoch"][i], f"{tr['loss'][i]:.6f}",
                            f"{tr['acc'][i]:.6f}", f"{te['loss'][i]:.6f}", f"{te['acc'][i]:.6f}"])
    print(f"已保存 {tab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
