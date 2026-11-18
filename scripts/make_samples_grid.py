# -*- coding: utf-8 -*-
"""从 MNIST 原始 idx 文件中随机抽取样本，渲染成「每行一类」的示例网格。

用途：PPT 第 6 页「MNIST 长什么样」，也可用于确认数据内容。
不依赖 torch，只读 gzip + numpy。

用法：
    python scripts/make_samples_grid.py
输出：
    docs/mnist_samples.png
"""
import gzip
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

RAW = os.path.join("data", "MNIST", "raw")
OUT = os.path.join("docs", "mnist_samples.png")
PER_CLASS = 10
SEED = 42


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError("bad magic: {}".format(magic))
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError("bad magic: {}".format(magic))
        return np.frombuffer(f.read(n), dtype=np.uint8)


def main():
    images = read_idx_images(os.path.join(RAW, "train-images-idx3-ubyte.gz"))
    labels = read_idx_labels(os.path.join(RAW, "train-labels-idx1-ubyte.gz"))
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(10, PER_CLASS, figsize=(PER_CLASS * 0.78, 10 * 0.86))
    fig.suptitle("MNIST 训练集示例：每行一个类别，每类 {} 个不同书写者的样本".format(PER_CLASS),
                 fontsize=14, fontweight="bold", y=0.995)

    for cls in range(10):
        pool = np.flatnonzero(labels == cls)
        picks = rng.choice(pool, size=PER_CLASS, replace=False)
        for j, idx in enumerate(picks):
            ax = axes[cls][j]
            ax.imshow(images[idx], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[cls][0].set_ylabel(str(cls), fontsize=15, fontweight="bold",
                                rotation=0, labelpad=14, va="center")

    fig.subplots_adjust(left=0.05, right=0.995, top=0.955, bottom=0.005,
                        wspace=0.06, hspace=0.06)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print("saved:", OUT)

    # 附带统计
    cnt = np.bincount(labels, minlength=10)
    print("total images :", images.shape[0])
    print("shape        :", images.shape[1:])
    print("pixel range  :", int(images.min()), "-", int(images.max()))
    print("labels       :", sorted(set(labels.tolist())))
    print("per class    :", cnt.tolist())


if __name__ == "__main__":
    main()
