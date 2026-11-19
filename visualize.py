# -*- coding: utf-8 -*-
"""组员 5 的统一绘图入口（对应《分工细则》§7.1 里的 ``visualize.py``）。

分工细则把 ``visualize.py`` 记为组员 5 维护的文件，这里作为**唯一入口**，
把四件事串起来；具体实现拆在 scripts/ 下，便于单步调试与复用。

用法
----
    python visualize.py                 # 全量：曲线 + 每类准确率/误判 + 核与特征 + 汇总表
    python visualize.py curve           # 只画学习曲线（fig1）
    python visualize.py errors          # 只做每类准确率与误判（fig2/6/7/8 + analysis/）
    python visualize.py features        # 只画卷积核与中间特征（fig3/4/5/9）
    python visualize.py summary         # 只生成结果汇总表（analysis/results_table.md）
    python visualize.py all --index 115 # 指定特征图用的测试样本

所有产物都写到 figures/ 与 analysis/，输入只读 outputs/，不修改任何实验记录。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

STEPS = {
    "curve": ("fig1 学习曲线", ["plot_learning_curve.py"]),
    "errors": ("fig2/6/7/8 每类准确率与误判分析", ["analyze_errors.py"]),
    "features": ("fig3/4/5/9 卷积核与中间特征", ["plot_kernels_features.py"]),
    "summary": ("results_table.md 结果汇总", ["summarize.py"]),
}
ORDER = ["curve", "errors", "features", "summary"]


def run(step: str, index: int | None) -> bool:
    title, cmd = STEPS[step]
    args = [sys.executable, str(SCRIPTS / cmd[0])]
    if step == "features" and index is not None:
        args += ["--index", str(index)]
    print(f"\n[{step}] {title}", flush=True)
    r = subprocess.run(args, cwd=ROOT)
    if r.returncode != 0:
        print(f"  !! {step} 失败，退出码 {r.returncode}", flush=True)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="组员 5 统一绘图入口")
    ap.add_argument("what", nargs="?", default="all",
                    choices=["all", *ORDER], help="要生成哪一部分，默认 all")
    ap.add_argument("--index", type=int, default=877, help="特征图使用的测试集索引（默认 877）")
    args = ap.parse_args()

    todo = ORDER if args.what == "all" else [args.what]
    ok = all(run(s, args.index) for s in todo)

    print("\n" + "=" * 60, flush=True)
    if ok:
        print("完成。产物位置：figures/ 与 analysis/")
        print("下一步检查：打开 figures/ 看四张主图，核对 analysis/results_table.md 的数字出处。")
    else:
        print("有步骤失败，请按上面的报错逐项排查（多半是 data/ 里缺 MNIST）。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
