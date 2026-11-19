# -*- coding: utf-8 -*-
"""5-05 卷积核 + 中间特征图（从最终 CNN 权重取，不用教学示意替代）。

产出
----
figures/fig3_conv1_kernels.png     conv1 的 6 个 5×5 学到的核（权重，不是示意图）
figures/fig4_conv1_features.png    同一张测试图的 conv1_relu（6×28×28）与 pool1（6×14×14）
figures/fig5_conv2_features.png    conv2_relu（16×10×10）与 pool2（16×5×5）
analysis/feature_shapes.csv        四路特征的实际形状与取值范围（写图注用）

口径（H2/H4 交接要求）
----------------------
- 键名固定为 conv1_relu / pool1 / conv2_relu / pool2，均为"卷积后 + ReLU 后"或"池化后"；
- 同一层内统一色标，不跨层比较绝对响应（不同层量纲不同）；
- 输入是**标准化后**的张量，与训练时完全一致；展示原图用 0—255 的 raw 图。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mnist_raw import load_test  # noqa: E402
from src.data import get_dataset_by_split  # noqa: E402
from src.models import load_model_from_checkpoint  # noqa: E402

CKPT = ROOT / "outputs" / "main_seed42_v1" / "checkpoints" / "cnn_seed42_last.pt"
OUT_F = ROOT / "figures"
OUT_A = ROOT / "analysis"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=877,
                    help="测试集索引（默认 877：一个 FC 判错、CNN 判对的 8）")
    args = ap.parse_args()
    idx = args.index

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"

    model, meta = load_model_from_checkpoint(CKPT)
    print("权重:", meta["model"], "seed", meta["seed"], "epoch", meta["epoch"])

    # 标准化输入（与训练完全一致）与 raw 原图（用于展示）
    x_norm, y = get_dataset_by_split("test")[idx]
    imgs_raw, labels = load_test()
    assert labels[idx] == y, "raw 与 torchvision 的索引对不上！"

    with torch.no_grad():
        feats = model.forward_features(x_norm.unsqueeze(0))
    feats = {k: v[0].numpy() for k, v in feats.items()}

    # ---------- 形状与取值 ----------
    OUT_A.mkdir(exist_ok=True)
    with open(OUT_A / "feature_shapes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["key", "shape", "min", "max", "mean"])
        for k, v in feats.items():
            w.writerow([k, "x".join(map(str, v.shape)),
                        f"{v.min():.4f}", f"{v.max():.4f}", f"{v.mean():.4f}"])
    print({k: v.shape for k, v in feats.items()})

    # ---------- 图 3：conv1 的 6 个核 ----------
    kernels = model.conv1.weight.detach().numpy()[:, 0]      # [6,5,5]
    vmax = float(np.abs(kernels).max())
    fig, axes = plt.subplots(1, 6, figsize=(9.6, 2.25), dpi=240)
    for c, ax in enumerate(axes):
        im = ax.imshow(kernels[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"通道 {c}\n5×5", fontsize=8.5)
        for s in ax.spines.values():
            s.set_color("#BBB")
    cax = fig.add_axes([0.22, 0.05, 0.56, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("学到的权重值（红=正，蓝=负）", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Conv1 的 6 个 5×5 卷积核（来自 cnn_seed42_last.pt，非教学示意）",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.80, bottom=0.28)
    out3 = OUT_F / "fig3_conv1_kernels.png"
    fig.savefig(out3, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 4 / 图 5：同一张图的中间激活 ----------
    def grid(layer_a: str, layer_b: str, name: str, title: str, note: str,
             with_raw: bool = False) -> Path:
        """with_raw=True 时最左列加一张原图，满足 §6.3「固定样本原图与指定层激活」同图呈现。"""
        a, b = feats[layer_a], feats[layer_b]
        n = max(a.shape[0], b.shape[0])
        off = 1 if with_raw else 0
        ncols = n + off
        kwargs = {"gridspec_kw": {"width_ratios": [1.3] + [1.0] * (ncols - 1)}} if with_raw else {}
        fig, axes = plt.subplots(2, ncols, figsize=(ncols * 1.06, 3.3), dpi=240, **kwargs)
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")

        if with_raw:
            ax = axes[0][0]
            ax.axis("on")
            ax.imshow(imgs_raw[idx], cmap="gray_r", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"原图 0—255\n真值 {y}", fontsize=7.6, pad=1.8)
            for s in ax.spines.values():
                s.set_color("#999")
                s.set_linewidth(0.8)
            axes[0][0].set_ylabel("模型输入", fontsize=8, color="#333")

        for row, (arr, layer) in enumerate([(a, layer_a), (b, layer_b)]):
            # 同层统一色标（用该层的最大值），不逐图归一化，避免"响应强弱"被拉平
            vmax = float(arr.max()) if arr.max() > 0 else 1.0
            for c in range(arr.shape[0]):
                ax = axes[row][c + off]
                ax.axis("on")
                ax.imshow(arr[c], cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                h, w_ = arr.shape[1], arr.shape[2]
                ax.set_title(f"ch{c} {h}×{w_}", fontsize=6.6, pad=1.5)
                for s in ax.spines.values():
                    s.set_color("#DDD")
                    s.set_linewidth(0.6)
            axes[row][off].set_ylabel(f"{layer}\n" + ("卷积+ReLU" if "conv" in layer else "最大池化"),
                                      fontsize=8, color="#333")
        fig.suptitle(title, fontsize=11.5, fontweight="bold", y=1.0)
        fig.text(0.5, -0.02, note, ha="center", fontsize=7.6, color="#666")
        fig.tight_layout()
        out = OUT_F / name
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    base = (f"输入：测试集第 {idx} 张（真值 {y}），已按 mean=0.1307 / std=0.3081 标准化；"
            "色标按层统一（inferno，0 为无响应），不跨层比较绝对强弱")
    out4 = grid("conv1_relu", "pool1", "fig4_conv1_features.png",
                "同一张图：原图 → Conv1 激活 → 第一次池化", base, with_raw=True)
    out5 = grid("conv2_relu", "pool2", "fig5_conv2_features.png",
                "同一张图的 Conv2 激活与第二次池化", base)

    # ---------- 附：原图 + 归一化后像素分布，放在同一张对照图里 ----------
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), dpi=240)
    axes[0].imshow(imgs_raw[idx], cmap="gray_r", interpolation="nearest")
    axes[0].set_title(f"原图 0—255（真值 {y}）", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(x_norm[0].numpy(), cmap="gray_r", interpolation="nearest")
    axes[1].set_title("标准化后（模型真实输入）", fontsize=9)
    axes[1].axis("off")
    axes[2].hist(x_norm[0].numpy().ravel(), bins=40, color="#3B6FB6")
    axes[2].set_title("标准化后的像素分布", fontsize=9)
    axes[2].set_xlabel("像素值")
    axes[2].tick_params(labelsize=7)
    for s in ("top", "right"):
        axes[2].spines[s].set_visible(False)
    fig.suptitle("同一张图：原图 vs 模型输入", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    out6 = OUT_F / "fig9_input_normalization.png"
    fig.savefig(out6, bbox_inches="tight")
    plt.close(fig)

    for p in (out3, out4, out5, out6):
        print("saved", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
