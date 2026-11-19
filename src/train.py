# -*- coding: utf-8 -*-
"""通用训练入口（组员 4）。

一份循环同时训练 FC 与 CNN，两个模型只差 ``--model`` 一个参数，
其余数据、超参、随机种子、评估口径完全共用，保证对比实验的控制变量。

用法
----
    python -m src.train --model fc  --config config.json --run-id main_seed42_v1
    python -m src.train --model cnn --config config.json --run-id main_seed42_v1

两个模型必须用**同一个 run-id**，它们才属于同一次对照实验。
再次运行同一 run-id 时，``metrics.csv`` 采用追加写入，不会覆盖已有轮次记录。

产出（``outputs/<run_id>/``）
---------------------------
    config.json            本次运行的配置快照
    environment.txt        Python / torch / 设备信息
    metrics.csv            每轮、每个数据划分一行
    checkpoints/<model>_seed<seed>_last.pt   最后一轮权重（含 model/seed/epoch）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 允许 `python src/train.py` 与 `python -m src.train` 两种跑法
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402
from src import models as M  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: metrics.csv 的列顺序，与《总体方案》§4.3 一致
METRIC_FIELDS = [
    "run_id", "model", "seed", "epoch", "split",
    "n_samples", "loss", "accuracy", "train_seconds",
]


# ---------------------------------------------------------------- 基础设施


def set_seed(seed: int) -> None:
    """固定所有随机源：python / numpy / torch。

    数据打乱另有独立 generator（见 ``data.make_generator``），
    不受这里的全局状态影响，因此两个模型的样本顺序严格一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name: str = "auto") -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tcfg = cfg["train"]
    name = str(tcfg.get("optimizer", "adam")).lower()
    lr = float(tcfg.get("lr", 1e-3))
    wd = float(tcfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"不支持的优化器：{name}")


def write_csv_row(path: Path, row: dict, fields: list[str]) -> None:
    """追加一行。文件不存在时先写表头，保证多次运行不互相覆盖。"""
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def dump_environment(path: Path, device: torch.device) -> None:
    import torchvision

    lines = [
        f"python      {sys.version.split()[0]}",
        f"executable  {sys.executable}",
        f"platform    {platform.platform()}",
        f"torch       {torch.__version__}",
        f"torchvision {torchvision.__version__}",
        f"numpy       {np.__version__}",
        f"device      {device}",
        f"cuda avail  {torch.cuda.is_available()}",
        f"threads     {torch.get_num_threads()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 评估


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn, device: torch.device) -> dict:
    """返回 {loss, accuracy, n_samples}，口径与《总体方案》§4.3 一致。

    - loss：先按 ``loss.item() * batch_size`` 累加，最后除以总样本数。
      不能对各 batch 的均值再取平均——末尾 batch 只有 32/16 张时会被过度加权。
    - accuracy：总正确数 / 总样本数。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        n = y.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += n
    if total == 0:
        raise RuntimeError("评估集为空")
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "n_samples": total,
    }


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="CNN/MNIST 通用训练入口（组员 4）")
    ap.add_argument("--model", required=True, choices=["fc", "cnn"], help="训练哪个模型")
    ap.add_argument("--config", default="config.json", help="配置文件路径")
    ap.add_argument("--run-id", default=None, help="实验编号，默认取 config 里的 run_id")
    ap.add_argument("--epochs", type=int, default=None,
                    help="调试用：临时覆盖轮次。正式实验不要用，会破坏与 config 的一致性")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_id = args.run_id or cfg.get("run_id", "run")
    seed = int(cfg["seed"])
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    if args.epochs is not None:
        print(f"[警告] 用 --epochs {epochs} 覆盖了 config 中的 {train_cfg['epochs']}，"
              f"这份结果不能作为正式实验")

    device = pick_device(train_cfg.get("device", "auto"))
    run_dir = PROJECT_ROOT / cfg["outputs"]["root"] / run_id
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- 配置快照与环境记录 ----
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = cfg_path.read_text(encoding="utf-8")
        if old != Path(args.config).read_text(encoding="utf-8"):
            alt = run_dir / f"config_{args.model}_{int(time.time())}.json"
            alt.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[警告] 配置与本次 run 已有快照不同，另存为 {alt.name}")
    else:
        cfg_path.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    dump_environment(run_dir / "environment.txt", device)

    print(f"run_id   {run_id}   model {args.model}   seed {seed}   device {device}")
    print(f"epochs   {epochs}   batch {data_cfg['batch_size']}   lr {train_cfg['lr']}")

    # ---- 固定种子后再建模型：两个模型才会从相同的初始分布出发 ----
    set_seed(seed)

    train_loader, train_eval_loader, test_loader = D.get_loaders(
        batch_size=int(data_cfg["batch_size"]),
        seed=seed,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        download=bool(data_cfg.get("download", False)),
    )
    print(f"data     train {len(train_loader.dataset)} 张 / "
          f"{len(train_loader)} batch   test {len(test_loader.dataset)} 张")

    model = M.build_model(args.model).to(device)
    n_params = M.count_parameters(model)
    expected = cfg["models"][args.model]["expected_params"]
    flag = "OK" if n_params == expected else f"MISMATCH(期望 {expected})"
    print(f"model    {args.model}  参数 {n_params:,}  {flag}")

    optimizer = build_optimizer(model, cfg)
    loss_fn = nn.CrossEntropyLoss()

    metrics_path = run_dir / "metrics.csv"
    history = []

    for epoch in range(1, epochs + 1):
        # ---- 训练一轮 ----
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()          # 必须：PyTorch 的梯度是累加的
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        train_sec = time.time() - t0

        # ---- 轮末统一评估（同一套 evaluate，两个划分口径一致）----
        tr = evaluate(model, train_eval_loader, loss_fn, device)
        te = evaluate(model, test_loader, loss_fn, device)

        for split, m in (("train", tr), ("test", te)):
            write_csv_row(metrics_path, {
                "run_id": run_id, "model": args.model, "seed": seed,
                "epoch": epoch, "split": split,
                "n_samples": m["n_samples"],
                "loss": f"{m['loss']:.6f}",
                "accuracy": f"{m['accuracy']:.6f}",
                "train_seconds": f"{train_sec:.1f}" if split == "train" else "",
            }, METRIC_FIELDS)

        history.append((epoch, train_sec, tr["accuracy"], te["accuracy"], te["loss"]))
        print(f"epoch {epoch}/{epochs}  {train_sec:6.1f}s  "
              f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f}  |  "
              f"test loss {te['loss']:.4f} acc {te['accuracy']:.4f}")

    # ---- 保存最后一轮（不从测试集挑最好的一轮）----
    ckpt_name = cfg["outputs"]["checkpoint_pattern"].format(model=args.model, seed=seed)
    ckpt_path = ckpt_dir / Path(ckpt_name).name
    M.save_checkpoint(
        model, ckpt_path, args.model, seed=seed, epoch=epochs,
        extra={"n_params": n_params, "run_id": run_id, "epochs": epochs,
               "batch_size": int(data_cfg["batch_size"]), "lr": float(train_cfg["lr"])},
    )
    print(f"saved    {ckpt_path}  ({ckpt_path.stat().st_size / 1024:.0f} KB)")

    # ---- 保存后立刻用新实例加载，确认权重真的可用 ----
    fixed_x = next(iter(test_loader))[0][:8].to(device)
    model.eval()
    with torch.no_grad():
        ref = model(fixed_x)
    reloaded, meta = M.load_model_from_checkpoint(ckpt_path, device=device)
    with torch.no_grad():
        got = reloaded(fixed_x)
    diff = (ref - got).abs().max().item()
    print(f"reload   {meta['model']} seed={meta['seed']} epoch={meta['epoch']}  "
          f"max|diff|={diff:.2e}  {'OK' if diff < 1e-6 else 'MISMATCH'}")

    best = max(history, key=lambda h: h[3]) if history else None
    if best:
        print(f"\n最后: test acc {history[-1][3]:.4f} / loss {history[-1][4]:.4f}")
        print(f"全程最好: epoch {best[0]} test acc {best[3]:.4f}（仅报告，不改变选用最后一轮）")
    print(f"日志     {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
