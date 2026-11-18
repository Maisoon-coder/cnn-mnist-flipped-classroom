# -*- coding: utf-8 -*-
"""从 MNIST 训练集里固定抽取一个分层子集，生成可复用的索引文件。

为什么单独存索引，而不是复制图片：
- 两个模型（FC / CNN）必须用**完全相同的样本**，否则对比不公平；
- 索引文件很小（几十 KB），可以进仓库；图片不用进仓库；
- 谁想知道"到底用了哪 3 万张"，打开这个文件就能复现。

用法：
    python scripts/make_subset.py                 # 默认 30000 张，seed=42
    python scripts/make_subset.py --n 20000 --seed 42
    python scripts/make_subset.py --n 60000       # 等于全量

输出：
    splits/train_<N>_seed<SEED>.npy   # numpy int64 索引，升序（进仓库，保证可复现）
"""
import argparse
import gzip
import os

import numpy as np

RAW = os.path.join("data", "MNIST", "raw")
LABEL_GZ = os.path.join(RAW, "train-labels-idx1-ubyte.gz")
OUT_DIR = "splits"


def read_train_labels():
    """读取 idx1 格式的标签文件，返回长度为 60000 的 uint8 数组。"""
    with gzip.open(LABEL_GZ, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError("bad magic number: {} (expect 2049)".format(magic))
        labels = np.frombuffer(f.read(n), dtype=np.uint8)
    if labels.shape[0] != n:
        raise ValueError("label file truncated")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="subset size, default 30000")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = read_train_labels()
    total = labels.shape[0]
    n = args.n
    if not 0 < n <= total:
        raise SystemExit("--n must be in 1..{}, got {}".format(total, n))

    rng = np.random.default_rng(args.seed)
    classes = np.unique(labels)
    per_class = n // len(classes)

    print("total train samples : {}".format(total))
    print("target subset size  : {}".format(n))
    print("classes             : {}".format(list(classes)))
    print("per class           : {}".format(per_class))

    picked = []
    for c in classes:
        pool = np.flatnonzero(labels == c)
        take = min(per_class, pool.size)
        picked.append(rng.choice(pool, size=take, replace=False))

    idx = np.sort(np.concatenate(picked)).astype(np.int64)
    # 若 n 不能被 10 整除，再从剩余样本里补足
    if idx.size < n:
        rest = np.setdiff1d(np.arange(total), idx, assume_unique=False)
        extra = rng.choice(rest, size=n - idx.size, replace=False)
        idx = np.sort(np.concatenate([idx, extra])).astype(np.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "train_{}_seed{}.npy".format(n, args.seed))
    np.save(out, idx)

    counts = np.bincount(labels[idx], minlength=len(classes))
    print()
    print("class distribution  : {}".format(dict(enumerate(counts.tolist()))))
    print("per-class max diff  : {}".format(int(counts.max() - counts.min())))
    print("saved               : {}  ({} indices, {} bytes)".format(
        out, idx.size, os.path.getsize(out)))
    print()
    print("usage:")
    print('  idx = np.load("{}")'.format(out.replace("\\", "/")))
    print("  train_ds = torch.utils.data.Subset(full_train_ds, idx)")


if __name__ == "__main__":
    main()
