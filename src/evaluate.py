# -*- coding: utf-8 -*-
"""独立评估入口（组员 4），供组员 5 复评与作图使用。

它做三件事：
1. 从**保存下来的权重**重新计算测试集指标（不读训练日志里的数字，避免"日志写错了也照样信"）；
2. 生成 ``results.csv``：每个模型一行的汇总；
3. 生成 ``errors_<model>.csv``：全部误判样本（测试集索引、真实标签、预测标签、概率），
   组员 5 据此画误判网格、回查原图。

用法
----
    python -m src.evaluate --run-dir outputs/main_seed42_v1
    python -m src.evaluate --run-dir outputs/main_seed42_v1 --model cnn
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402
from src import models as M  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FIELDS = [
    "run_id", "model", "seed", "epoch", "n_params",
    "test_acc", "test_loss", "n_samples", "checkpoint",
]
ERROR_FIELDS = [
    "index", "true_label", "pred_label",
    "prob_true", "prob_pred", "margin",
]


def find_checkpoints(run_dir: Path, model: str | None) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"没有找到权重目录：{ckpt_dir}")
    files = sorted(ckpt_dir.glob("*.pt"))
    if model:
        files = [f for f in files if model in f.name]
    if not files:
        raise SystemExit(f"{ckpt_dir} 下没有匹配 {'*' if not model else model} 的权重")
    return files


@torch.no_grad()
def eval_with_details(model: nn.Module, loader, loss_fn, device) -> dict:
    """逐样本评估，返回指标 + 每个样本的预测、概率、真实标签。

    ``loader`` 必须 **不打乱**，这样 ``running_index`` 才等于数据集原始索引，
    组员 5 才能用索引回查原图。
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    idx_list: list[int] = []
    true_list: list[int] = []
    pred_list: list[int] = []
    p_true_list: list[float] = []
    p_pred_list: list[float] = []

    running_index = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)          # 仅用于展示，不参与训练
        pred = probs.argmax(dim=1)
        n = y.size(0)

        total_loss += loss.item() * n
        correct += (pred == y).sum().item()
        total += n

        idx_list.extend(range(running_index, running_index + n))
        running_index += n
        true_list.extend(y.tolist())
        pred_list.extend(pred.tolist())
        p_pred_list.extend(probs.gather(1, pred.view(-1, 1)).squeeze(1).tolist())
        p_true_list.extend(probs.gather(1, y.view(-1, 1)).squeeze(1).tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "n_samples": total,
        "index": idx_list,
        "true": true_list,
        "pred": pred_list,
        "prob_true": p_true_list,
        "prob_pred": p_pred_list,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="独立评估（组员 4）")
    ap.add_argument("--run-dir", required=True, help="例如 outputs/main_seed42_v1")
    ap.add_argument("--model", default=None, choices=["fc", "cnn"], help="只评估某一个模型")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"目录不存在：{run_dir}")

    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    seed = cfg.get("seed", 42)
    run_id = cfg.get("run_id", run_dir.name)
    data_root = cfg.get("data", {}).get("root", "data")

    device = torch.device(args.device)
    # 测试加载器：不打乱，索引与数据集一致
    test_loader = D.get_test_loader(
        batch_size=args.batch_size, num_workers=0, data_root=data_root,
        download=False,
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"run_id {run_id}   test {len(test_loader.dataset)} 张   device {device}")

    rows = []
    for ckpt in find_checkpoints(run_dir, args.model):
        model, meta = M.load_model_from_checkpoint(ckpt, device=device)
        name = meta.get("model")
        n_params = M.count_parameters(model)
        r = eval_with_details(model, test_loader, loss_fn, device)

        print(f"\n[{name}] acc {r['accuracy']:.4f}  loss {r['loss']:.4f}  "
              f"params {n_params:,}  (seed={meta.get('seed')} epoch={meta.get('epoch')})")

        # 每类准确率，便于组员 5 分析"哪一类最容易错"
        per_class_correct = [0] * 10
        per_class_total = [0] * 10
        for t, p in zip(r["true"], r["pred"]):
            per_class_total[t] += 1
            if t == p:
                per_class_correct[t] += 1
        acc_line = "  ".join(
            f"{c}:{(per_class_correct[c] / per_class_total[c] if per_class_total[c] else 0):.3f}"
            for c in range(10))
        print("  每类准确率 " + acc_line)

        rows.append({
            "run_id": run_id, "model": name, "seed": meta.get("seed"),
            "epoch": meta.get("epoch"), "n_params": n_params,
            "test_acc": f"{r['accuracy']:.6f}", "test_loss": f"{r['loss']:.6f}",
            "n_samples": r["n_samples"],
            "checkpoint": str(ckpt.relative_to(run_dir)).replace("\\", "/"),
        })

        # 误判清单
        err_path = run_dir / f"errors_{name}.csv"
        n_err = 0
        with open(err_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS)
            writer.writeheader()
            for i, t, p, pt, pp in zip(r["index"], r["true"], r["pred"],
                                       r["prob_true"], r["prob_pred"]):
                if t == p:
                    continue
                writer.writerow({
                    "index": i, "true_label": t, "pred_label": p,
                    "prob_true": f"{pt:.6f}", "prob_pred": f"{pp:.6f}",
                    "margin": f"{pp - pt:.6f}",
                })
                n_err += 1
        print(f"  误判 {n_err} 个 -> {err_path.name}")

    if rows:
        out = run_dir / "results.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n汇总 -> {out}")
        print()
        print(f"{'model':<6}{'acc':>9}{'loss':>10}{'params':>10}")
        for row in rows:
            print(f"{row['model']:<6}{float(row['test_acc']):>9.4f}"
                  f"{float(row['test_loss']):>10.4f}{row['n_params']:>10,}")

        # 一句可直接抄进 PPT 的对比
        accs = {r["model"]: float(r["test_acc"]) for r in rows}
        if "fc" in accs and "cnn" in accs:
            gap = (accs["cnn"] - accs["fc"]) * 100
            drop = (1 - 61706 / 203530) * 100
            print(f"\nCNN 比 FC 准确率高 {gap:+.2f} 个百分点，参数少 {drop:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
