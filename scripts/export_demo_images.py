"""从 MNIST 测试集导出演示用单图（组员 3 的演示样本 + 来源说明）。

在项目根目录执行：

    python scripts/export_demo_images.py

导出规则：取测试集中**每个数字第一次出现**的样本，存成 ``demo/digit<标签>_test<索引>.png``，
并写出 ``demo/来源说明.md``，记录每张图的测试集索引。这样 PPT 上每张演示图都能追溯
到具体样本，不会被质疑"图是挑过的"。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from src.data import get_raw_dataset_by_split

DEMO_DIR = PROJECT_ROOT / "demo"


def main() -> int:
    dataset = get_raw_dataset_by_split("test")     # 只做 ToTensor，保留 0—1 原始灰度
    DEMO_DIR.mkdir(exist_ok=True)

    found: dict[int, int] = {}
    for index in range(len(dataset)):
        image, label = dataset[index]
        label = int(label)
        if label in found:
            continue
        found[label] = index
        path = DEMO_DIR / f"digit{label}_test{index}.png"
        pixels = (image.squeeze(0) * 255).round().to(torch.uint8).numpy()
        # 保存为 8 位灰度 PNG：与 MNIST 原始 0—255 存储一致，读回后预处理结果不变
        Image.fromarray(pixels, mode="L").save(path)
        print(f"已导出 digit{label} → {path.name}")
        if len(found) == 10:
            break

    lines = [
        "# 演示样本来源说明",
        "",
        "全部样本来自 MNIST 官方**测试集**（与训练集无重叠），由 `scripts/export_demo_images.py` 导出。",
        "导出规则：每个数字取测试集中**第一次出现**的样本，未做任何挑选或人工修图。",
        "",
        "| 文件 | 真实标签 | 测试集索引 | 用途 |",
        "|---|---|---|---|",
    ]
    for label in sorted(found):
        index = found[label]
        lines.append(f"| digit{label}_test{index}.png | {label} | {index} | 单图预测演示 / 错误分析对照 |")
    lines += [
        "",
        "## 使用注意",
        "",
        "- 这些图是 28×28、黑底白字，与训练分布**完全一致**，`predict.py` 不需要 `--invert`。",
        "- 若另外拍摄手写数字（白底黑字、非 28×28），必须加 `--invert`，并注意缩放与居中，",
        "  这类图片只作附加展示，不能用来代替测试集指标。",
        "- 现场演示优先用上表中的图片，可复现、可追溯。",
    ]
    (DEMO_DIR / "来源说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {DEMO_DIR / '来源说明.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
