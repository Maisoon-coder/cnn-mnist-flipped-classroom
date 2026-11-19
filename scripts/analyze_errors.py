# -*- coding: utf-8 -*-
"""5-06 错误分析 + 5-07 结果汇总：只用 errors_*.csv 与 MNIST 原图，不依赖 torch。

产出
----
analysis/per_class_accuracy.csv   每类准确率（分母用**真实**每类样本数，不是 1000）
analysis/error_pairs.csv          两模型的混淆三元组与次数
analysis/error_summary.md         可直接粘进讲稿的结论与口径说明
figures/fig2_per_class_accuracy.png
figures/fig6_error_grid_fc.png
figures/fig7_error_grid_cnn.png
figures/fig8_fc_wrong_cnn_right.png

注意：MNIST 测试集**每类不是 1000 张**（如 5 只有 892 张、1 有 1135 张），
算每类准确率必须用真实分母，否则会和组员 4 的数字差 0.2—0.4 个百分点。
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_raw import load_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "main_seed42_v1"
OUT_A = ROOT / "analysis"
OUT_F = ROOT / "figures"

COLOR = {"fc": "#3B6FB6", "cnn": "#D9662B"}
NAME = {"fc": "FC（全连接）", "cnn": "CNN（卷积）"}


def read_errors(model: str) -> list[dict]:
    with open(RUN / f"errors_{model}.csv", encoding="utf-8-sig") as f:
        return [
            {"index": int(r["index"]), "true": int(r["true_label"]),
             "pred": int(r["pred_label"]), "p_true": float(r["prob_true"]),
             "p_pred": float(r["prob_pred"])}
            for r in csv.DictReader(f)
        ]


def style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def fig_per_class(tot, errs) -> Path:
    """每类准确率分组柱状图：8 是最值得讲的一类。"""
    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=220)
    x = np.arange(10)
    w = 0.36
    for k, m in enumerate(("fc", "cnn")):
        accs = [(tot[c] - errs[m][c]) / tot[c] * 100 for c in range(10)]
        off = -w / 2 if k == 0 else w / 2
        bars = ax.bar(x + off, accs, w, color=COLOR[m], label=NAME[m],
                      edgecolor="white", linewidth=0.6)
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.12, f"{a:.1f}",
                    ha="center", va="bottom", fontsize=7.6, color="#333")

    # 数字 8 的差距单独框出来
    i = 8
    fc8 = (tot[8] - errs["fc"][8]) / tot[8] * 100
    cnn8 = (tot[8] - errs["cnn"][8]) / tot[8] * 100
    ax.annotate(f"8：差 {cnn8 - fc8:.1f} 个百分点\n（最大单项差距）",
                xy=(i - 0.18, fc8 + 0.6), xytext=(5.55, 89.35), fontsize=8.6, color="#333",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#777"),
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFF8EF", ec="#E0B98A", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel("真实类别（0—9）")
    ax.set_ylabel("该类准确率（%）")
    ax.set_ylim(89, 102.5)
    ax.set_yticks(range(90, 102, 2))
    ax.grid(axis="y", ls=":", lw=0.8, color="#C8CCD2")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # 图例放在坐标区上方，避免压住柱子
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    ax.set_title("每类准确率：FC 与 CNN（MNIST 测试集，seed=42）",
                 fontsize=12.5, fontweight="bold", pad=22)
    fig.text(0.5, 0.005,
             "分母为该类真实样本数（测试集各类并不都是 1000 张：5 类 892 张、1 类 1135 张）",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = OUT_F / "fig2_per_class_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _grid(imgs, rows: list[dict], title: str, out: Path, cols: int = 6, note: str = "") -> Path:
    n = min(len(rows), 24)
    r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(cols * 1.55, r * 1.85), dpi=220)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, e in zip(axes.ravel(), rows[:n]):
        ax.imshow(imgs[e["index"]], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"真 {e['true']} / 预 {e['pred']}\nidx {e['index']}", fontsize=8.0,
                     color="#B3261E", fontweight="bold", pad=3.0)
    fig.suptitle(title, fontsize=12.5, fontweight="bold")
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_same_samples(imgs, fc_ok_pairs, out: Path) -> Path:
    """同一张图：上排 FC 判错、下排 CNN 判对——排除"图本身难"的解释。"""
    n = len(fc_ok_pairs)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 4.4), dpi=220)
    for i, e in enumerate(fc_ok_pairs):
        axes[0][i].imshow(imgs[e["index"]], cmap="gray_r", interpolation="nearest")
        axes[0][i].set_title(f"FC 判 {e['pred']}（真 {e['true']}）", fontsize=8.2,
                             color="#B3261E", fontweight="bold")
        axes[1][i].imshow(imgs[e["index"]], cmap="gray_r", interpolation="nearest")
        axes[1][i].set_title(f"CNN 判 {e['true']}（正确）", fontsize=8.2,
                             color="#1B6B3A", fontweight="bold")
        axes[1][i].set_xlabel(f"idx {e['index']}", fontsize=7.2, color="#777", labelpad=2)
        for row in (0, 1):
            axes[row][i].set_xticks([])
            axes[row][i].set_yticks([])
            for s in axes[row][i].spines.values():
                s.set_visible(True)
                s.set_linewidth(1.2)
                s.set_color("#B3261E" if row == 0 else "#1B6B3A")
    axes[0][0].set_ylabel("FC", fontsize=11, fontweight="bold", color="#B3261E")
    axes[1][0].set_ylabel("CNN", fontsize=11, fontweight="bold", color="#1B6B3A")
    fig.suptitle("同一批图片：FC 判错、CNN 判对（以 8 为主）", fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.012,
             "图片、预处理完全相同，差别只在模型结构——因此不能把这批图简单归因为“字写得太差”",
             ha="center", fontsize=8, color="#666")
    fig.subplots_adjust(left=0.05, right=0.995, top=0.86, bottom=0.10, hspace=0.42, wspace=0.08)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    OUT_A.mkdir(exist_ok=True)
    OUT_F.mkdir(exist_ok=True)
    style()

    imgs, labels = load_test()
    tot = np.bincount(labels, minlength=10)

    errs = {m: read_errors(m) for m in ("fc", "cnn")}
    cnt = {m: Counter(e["true"] for e in errs[m]) for m in ("fc", "cnn")}

    # ---------- 每类准确率 ----------
    with open(OUT_A / "per_class_accuracy.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["digit", "n_test", "fc_errors", "fc_acc", "cnn_errors", "cnn_acc", "gap_pp"])
        for c in range(10):
            fca = (tot[c] - cnt["fc"][c]) / tot[c] * 100
            cna = (tot[c] - cnt["cnn"][c]) / tot[c] * 100
            w.writerow([c, tot[c], cnt["fc"][c], f"{fca:.2f}", cnt["cnn"][c], f"{cna:.2f}",
                        f"{cna - fca:+.2f}"])

    # ---------- 混淆三元组 ----------
    pair_rows = []
    for m in ("fc", "cnn"):
        pc = Counter((e["true"], e["pred"]) for e in errs[m])
        for (t, p), n in pc.most_common():
            pair_rows.append([m, t, p, n])
    with open(OUT_A / "error_pairs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "true", "pred", "count"])
        w.writerows(pair_rows)

    # ---------- 图 ----------
    p2 = fig_per_class(tot, cnt)
    fc_sorted = sorted(errs["fc"], key=lambda e: (e["true"] != 8, -e["p_pred"]))
    cnn_sorted = sorted(errs["cnn"], key=lambda e: -e["p_pred"])
    p5 = _grid(imgs, fc_sorted, "FC 误判样本（优先展示 8；标“真 x / 预 y”与测试集索引）",
               OUT_F / "fig6_error_grid_fc.png",
               note="图片来源：MNIST t10k 原图（0—255），errors_fc.csv 的 index 直接索引，可逐张复查")
    p6 = _grid(imgs, cnn_sorted, "CNN 误判样本（标“真 x / 预 y”与测试集索引）",
               OUT_F / "fig7_error_grid_cnn.png",
               note="图片来源：MNIST t10k 原图（0—255），errors_cnn.csv 的 index 直接索引，可逐张复查")

    cnn_err_idx = {e["index"] for e in errs["cnn"]}
    fc_only = [e for e in errs["fc"] if e["index"] not in cnn_err_idx]
    eight = [e for e in fc_only if e["true"] == 8]
    pick = (eight + [e for e in fc_only if e["true"] != 8])[:8]
    pick.sort(key=lambda e: e["index"])
    p7 = fig_same_samples(imgs, pick, OUT_F / "fig8_fc_wrong_cnn_right.png")

    # ---------- 结论草稿 ----------
    fc8 = (tot[8] - cnt["fc"][8]) / tot[8] * 100
    cnn8 = (tot[8] - cnt["cnn"][8]) / tot[8] * 100
    fc8_pairs = Counter(e["pred"] for e in errs["fc"] if e["true"] == 8)
    cnn8_pairs = Counter(e["pred"] for e in errs["cnn"] if e["true"] == 8)
    fc_total = 10000 - len(errs["fc"])
    cnn_total = 10000 - len(errs["cnn"])

    md = f"""# 错误分析与结果汇总（组员 5，由脚本生成，勿手改数字）

## 1 总量
| 模型 | 正确/总数 | 准确率 | 误判条数 |
|---|---:|---:|---:|
| FC | {fc_total}/10000 | {fc_total/100:.2f}% | {len(errs['fc'])} |
| CNN | {cnn_total}/10000 | {cnn_total/100:.2f}% | {len(errs['cnn'])} |

差值：CNN 高 {(cnn_total-fc_total)/100:.2f} 个百分点。

## 2 数字 8（最值得讲的一类）
- FC：错 {cnt['fc'][8]}/{tot[8]}，该类准确率 {fc8:.2f}%；误判去向 {dict(fc8_pairs.most_common())}
- CNN：错 {cnt['cnn'][8]}/{tot[8]}，该类准确率 {cnn8:.2f}%；误判去向 {dict(cnn8_pairs.most_common())}
- 差距 {cnn8 - fc8:+.2f} 个百分点。

⚠️ 口径更正：H4 交接文档写“FC 在 8 上总共错了 71 个”“CNN 只错 5 个”，
71 是 8→9/8→3/8→5/8→4 四项之和（另有 {cnt['fc'][8]-71} 个分布在 8→7/0/2/6），
5 是 CNN 8→3 的单项数；真实总数为 FC {cnt['fc'][8]} 个、CNN {cnt['cnn'][8]} 个。

## 3 FC 错、CNN 对（同图对照）
- 仅 FC 错（CNN 对）的样本：{len(fc_only)} 个
- 其中真实为 8 的：{len(eight)} 个 → 用作第 14 页同图对照（fig7）

## 4 高频混淆
FC 前 5：{[(f'{t}→{p}', n) for t, p, n in [(int(r[1]), int(r[2]), int(r[3])) for r in pair_rows if r[0]=='fc'][:5]]}
CNN 前 5：{[(f'{t}→{p}', n) for t, p, n in [(int(r[1]), int(r[2]), int(r[3])) for r in pair_rows if r[0]=='cnn'][:5]]}
"""
    (OUT_A / "error_summary.md").write_text(md, encoding="utf-8")

    # 供 visualize.py 复用的统一入口（原始实验变化后重跑本脚本即可）
    for p in (p2, p5, p6, p7):
        print("saved", p.relative_to(ROOT))
    print("saved analysis/per_class_accuracy.csv, error_pairs.csv, error_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
