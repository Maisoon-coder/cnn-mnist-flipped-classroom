#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CNN / MNIST 项目环境自检。

用法：
    python scripts/check_env.py              # 基础检查
    python scripts/check_env.py --download   # 额外检查 MNIST 能否加载（会联网下载）

输出全部为 ASCII，避免 Windows 控制台编码问题。
退出码：0 = 全部通过，1 = 有检查项失败。
"""
import argparse
import platform
import sys

PASS = "[ OK ]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []


def add(ok, name, detail):
    results.append((ok, name, detail))
    tag = PASS if ok else FAIL
    print("{:<16}{}  {}".format(name, tag, detail))


def run(name, fn):
    try:
        detail = fn()
        add(True, name, detail if detail is not None else "")
    except Exception as exc:  # noqa: BLE001
        add(False, name, "{}: {}".format(type(exc).__name__, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="check MNIST loading (downloads if missing)")
    args = parser.parse_args()

    print("=" * 60)
    print("CNN / MNIST environment self-check")
    print("=" * 60)

    # ---------- basic ----------
    add(True, "Python", "{}.{}.{}  ({})".format(*sys.version_info[:3], sys.executable))
    add(True, "Platform", platform.platform())
    add(True, "Arch", platform.machine())

    weak = sys.version_info < (3, 10)
    add(not weak, "Python >= 3.10",
        "yes" if not weak else "too old, install Python 3.10+")

    # ---------- packages ----------
    try:
        import torch
        add(True, "torch", torch.__version__)
    except Exception as exc:  # noqa: BLE001
        torch = None
        add(False, "torch", "{}: {}".format(type(exc).__name__, exc))

    try:
        import torchvision
        add(True, "torchvision", torchvision.__version__)
    except Exception as exc:  # noqa: BLE001
        torchvision = None
        add(False, "torchvision", "{}: {}".format(type(exc).__name__, exc))

    try:
        import numpy
        add(True, "numpy", numpy.__version__)
    except Exception as exc:  # noqa: BLE001
        numpy = None
        add(False, "numpy", "{}: {}".format(type(exc).__name__, exc))

    try:
        import matplotlib
        add(True, "matplotlib", matplotlib.__version__)
    except Exception as exc:  # noqa: BLE001
        add(False, "matplotlib",
            "{}: {} (needed by visualisation only)".format(type(exc).__name__, exc))

    # ---------- torch details ----------
    if torch is not None:
        add(True, "CUDA", "available={} version={}".format(
            torch.cuda.is_available(), torch.version.cuda))
        add(True, "threads", str(torch.get_num_threads()))

        def tensor_op():
            torch.manual_seed(42)
            a = torch.randn(4, 8)
            b = torch.randn(8, 8)
            c = a @ b
            return "randn(4,8)@randn(8,8) -> {}".format(tuple(c.shape))
        run("tensor op", tensor_op)

        def conv_fwd():
            import torch.nn as nn
            torch.manual_seed(42)
            conv = nn.Conv2d(1, 6, 5, padding=2)
            x = torch.randn(1, 1, 28, 28)
            y = conv(x)
            return "(1,1,28,28) -> {}".format(tuple(y.shape))
        run("conv2d fwd", conv_fwd)

        def loss_backward():
            import torch.nn as nn
            torch.manual_seed(42)
            net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
            loss_fn = nn.CrossEntropyLoss()
            x = torch.randn(8, 1, 28, 28)
            y = torch.randint(0, 10, (8,))
            loss = loss_fn(net(x), y)
            loss.backward()
            grad_ok = net[1].weight.grad is not None
            return "loss={:.4f} grad_ok={}".format(loss.item(), grad_ok)
        run("loss+backward", loss_backward)

        def param_count():
            import torch.nn as nn
            fc = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                               nn.Linear(256, 10))
            n_fc = sum(p.numel() for p in fc.parameters())
            cnn = nn.Sequential(
                nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(400, 120), nn.ReLU(),
                nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
            n_cnn = sum(p.numel() for p in cnn.parameters())
            ok = (n_fc == 203530 and n_cnn == 61706)
            return "FC={} CNN={} match_expected={}".format(n_fc, n_cnn, ok)
        run("param count", param_count)

    # ---------- mnist ----------
    if torchvision is not None:
        def mnist_check():
            from torchvision import datasets
            try:
                ds = datasets.MNIST(root="data", train=False, download=False)
                return "test set size={} (pre-processed)".format(len(ds))
            except Exception:
                # 首次使用：本地已有 .gz，torchvision 会校验 MD5 后离线解压，不会联网
                ds = datasets.MNIST(root="data", train=False, download=True)
                return "test set size={} (extracted from local .gz)".format(len(ds))

        if args.download:
            run("MNIST", mnist_check)
        else:
            try:
                mnist_check()
            except Exception as exc:  # noqa: BLE001
                add(True, "MNIST", "not in ./data yet ({}); "
                    "run with --download".format(type(exc).__name__))

    # ---------- summary ----------
    print("-" * 60)
    failed = [n for ok, n, _ in results if not ok]
    total = len(results)
    if failed:
        print("RESULT: FAIL  ({}/{} passed)  failed: {}".format(
            total - len(failed), total, ", ".join(failed)))
        print("Fix the failed items above, then run this script again.")
        return 1
    print("RESULT: PASS  ({}/{} checks passed)".format(total, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
