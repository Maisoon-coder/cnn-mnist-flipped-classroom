# -*- coding: utf-8 -*-
"""不依赖 torch / torchvision 的最小 MNIST 读取（组员 5 作图用）。

为什么不用 ``datasets.MNIST``：本机是零依赖环境（没有 torch），
而误判网格只需要"第 i 张图长什么样"。直接读 idx 文件既快又不引入框架，
顺带证明"索引 → 原图"这条链路是通的（H4 交接文档强调的 5-06 要求）。
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw"


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic not in (0x00000801, 0x00000803):
            raise ValueError(f"{path.name} 不是 idx 文件（magic={magic:#x}）")
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        buf = f.read()
    dt = np.uint8
    arr = np.frombuffer(buf, dtype=dt)
    return arr.reshape(dims) if len(dims) > 1 else arr


def load_test() -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(images[10000,28,28] uint8 0—255, labels[10000] uint8)``。

    索引与 ``datasets.MNIST(train=False)`` 完全一致（都是 t10k 的原始顺序），
    所以 errors_*.csv 的 index 可以直接索引到这里的第 index 张图。
    """
    imgs = _read_idx(RAW_DIR / "t10k-images-idx3-ubyte.gz")
    labels = _read_idx(RAW_DIR / "t10k-labels-idx1-ubyte.gz")
    if imgs.shape[0] != 10000 or labels.shape[0] != 10000:
        raise ValueError(f"测试集样本数不对：{imgs.shape[0]} / {labels.shape[0]}")
    return imgs, labels


if __name__ == "__main__":
    x, y = load_test()
    print("images", x.shape, x.dtype, x.min(), x.max())
    print("labels", y.shape, np.bincount(y))
