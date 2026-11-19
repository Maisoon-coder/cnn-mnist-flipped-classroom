"""单图预测入口（组员 3 交付，兼现场演示用）。

用法
----
用 MNIST 格式的图片预测（推荐，预处理与训练集完全一致）：

    python -m src.predict --checkpoint outputs/main_seed42_v1/checkpoints/cnn_seed42_last.pt --image demo/digit_7_test0.png

直接按测试集索引预测（不落盘，用于核对权重是否正常）：

    python -m src.predict --checkpoint <权重> --mnist-index 0 --split test

手写/拍照图片（白底黑字，需要反转）：

    python -m src.predict --checkpoint <权重> --image my_digit.png --invert

关键点：预处理**只调用** ``src.data.build_transform()``，与训练同一份实现、同一组
mean/std。这样"训练看到的输入分布"和"预测看到的输入分布"必然一致，不会出现
演示时准确率莫名下降、却查不出原因的情况。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, ImageOps

# 允许用 `python src/predict.py` 直接跑（此时没有包上下文，退回绝对导入）
try:  # pragma: no cover - 只是导入方式兼容
    from .data import build_transform, get_dataset_by_split
    from .models import load_model_from_checkpoint
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import build_transform, get_dataset_by_split
    from src.models import load_model_from_checkpoint

MNIST_SIZE = (28, 28)


# ---------------------------------------------------------------- 预处理


def load_image_as_array(image_path: str | Path, invert: bool = False) -> Image.Image:
    """把任意图片读成"MNIST 口径"的灰度图。

    与训练分布对齐的三件事：
    1. **灰度**：训练只有 1 个通道，彩色图必须先转 ``L``。
    2. **极性**：MNIST 是黑底白字（0 = 背景）。白底黑字的照片需要 ``invert=True``。
    3. **尺寸 28×28**：先补成正方形再缩放，避免直接拉伸把数字压扁变形。

    如果原图已经是 28×28，这里不会做任何缩放/补边，保证与训练集逐像素一致。
    """
    img = Image.open(image_path).convert("L")
    if invert:
        img = ImageOps.invert(img)

    if img.size != MNIST_SIZE:
        # 补边到正方形：背景色取 0（黑），符合 MNIST 的"黑底"约定
        side = max(img.size)
        square = Image.new("L", (side, side), color=0)
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        img = square.resize(MNIST_SIZE, Image.LANCZOS)
    return img


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL 灰度图 → 模型输入张量 ``[1, 1, 28, 28]``。

    ``build_transform()`` 内含 ToTensor（0—255 → 0—1）与 Normalize（减均值除标准差），
    与训练集完全同一套参数。
    """
    tensor = build_transform()(image)      # [1,28,28]
    return tensor.unsqueeze(0)             # 补上 batch 维 → [1,1,28,28]


# ---------------------------------------------------------------- 推理


@torch.no_grad()   # 预测不需要梯度：省内存、更快，也避免误把推理当训练
def predict_tensor(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: str | torch.device = "cpu",
) -> Tuple[int, torch.Tensor]:
    """对 ``[1,1,28,28]`` 张量推理，返回 ``(预测类别, 10 类概率)``。

    概率只用于说明模型输出的相对大小，**不是**经过校准的置信度，
    答辩时不要说"模型 99.8% 确定"。
    """
    model.eval()
    logits = model(tensor.to(device))                    # [1,10]
    probs = torch.softmax(logits, dim=1)[0].cpu()         # 展示时才算 softmax
    return int(probs.argmax().item()), probs


def predict_image(
    checkpoint: str | Path,
    image_path: str | Path,
    model_name: Optional[str] = None,
    invert: bool = False,
    device: str | torch.device = "cpu",
) -> Tuple[int, torch.Tensor]:
    """从图片路径直接预测，供演示与其它模块调用。"""
    model, _ = load_model_from_checkpoint(checkpoint, model_name=model_name, device=device)
    image = load_image_as_array(image_path, invert=invert)
    return predict_tensor(model, preprocess_image(image), device=device)


# ---------------------------------------------------------------- 命令行


def _format_report(pred: int, probs: torch.Tensor, topk: int) -> str:
    lines = [f"预测结果：{pred}"]
    values, indices = torch.topk(probs, k=min(topk, probs.numel()))
    lines.append(f"{'类别':<6}{'概率':>10}")
    for value, index in zip(values.tolist(), indices.tolist()):
        lines.append(f"{index:<6}{value * 100:>9.2f}%")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MNIST 单图预测（组员 3）")
    parser.add_argument("--checkpoint", required=True, help="权重文件路径（.pt）")
    parser.add_argument("--image", help="待预测图片路径")
    parser.add_argument("--mnist-index", type=int, help="改用 MNIST 测试/训练集索引，免去准备图片")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="配合 --mnist-index 使用，默认 test")
    parser.add_argument("--model", help="模型名 fc/cnn；权重文件已记录时可省略")
    parser.add_argument("--invert", action="store_true",
                        help="反转灰度（白底黑字的照片需要加这个）")
    parser.add_argument("--topk", type=int, default=10, help="显示前 k 个类别概率，默认 10")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda，演示建议 cpu")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本读取")
    args = parser.parse_args(argv)

    if bool(args.image) == (args.mnist_index is not None):
        parser.error("--image 与 --mnist-index 必须且只能给一个")

    # 现场演示时不要甩一屏 traceback：权重缺失/格式不对就直接给一句能照做的提示
    try:
        model, meta = load_model_from_checkpoint(
            args.checkpoint, model_name=args.model, device=args.device
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[错误] 无法加载权重：{exc}", file=sys.stderr)
        print("检查：--checkpoint 路径是否存在；若权重里没记录模型名，加 --model fc/cnn。",
              file=sys.stderr)
        return 2

    if args.image:
        try:
            image = load_image_as_array(args.image, invert=args.invert)
        except FileNotFoundError:
            print(f"[错误] 找不到图片：{args.image}", file=sys.stderr)
            return 2
        except OSError as exc:            # 不是图片、文件被占用、格式不支持
            print(f"[错误] 图片无法读取：{args.image}（{exc}）", file=sys.stderr)
            return 2
        pred, probs = predict_tensor(model, preprocess_image(image), device=args.device)
        source = str(args.image)
        true_label = None
    else:
        dataset = get_dataset_by_split(args.split)
        if not 0 <= args.mnist_index < len(dataset):
            print(f"[错误] 索引 {args.mnist_index} 超出 {args.split} 集范围 "
                  f"(0—{len(dataset) - 1})", file=sys.stderr)
            return 2
        tensor, true_label = dataset[args.mnist_index]     # 已含标准化，与训练完全一致
        pred, probs = predict_tensor(model, tensor.unsqueeze(0), device=args.device)
        source = f"{args.split}[{args.mnist_index}]"

    if args.json:
        print(json.dumps(
            {
                "source": source,
                "checkpoint": str(args.checkpoint),
                "model": meta.get("model"),
                "seed": meta.get("seed"),
                "epoch": meta.get("epoch"),
                "prediction": pred,
                "true_label": int(true_label) if true_label is not None else None,
                "probabilities": [round(float(p), 6) for p in probs.tolist()],
            },
            ensure_ascii=False,
        ))
    else:
        print(f"权重：{args.checkpoint}（model={meta.get('model')}, seed={meta.get('seed')}, epoch={meta.get('epoch')}）")
        print(f"输入：{source}")
        if true_label is not None:
            print(f"真实标签：{true_label}")
        print(_format_report(pred, probs, args.topk))
        if true_label is not None:
            print("判断：" + ("正确" if int(true_label) == pred else "错误"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
