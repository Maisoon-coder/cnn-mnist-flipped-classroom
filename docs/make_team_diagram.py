# -*- coding: utf-8 -*-
"""生成《五人协作与汇报关系图》。依据：CNN任务总体方案.md、CNN五人分工与协作细则.md"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

C = {
    1: ("#E8F0FE", "#1A73E8"),
    2: ("#FCE8E6", "#D93025"),
    3: ("#E6F4EA", "#188038"),
    4: ("#FEF7E0", "#E37400"),
    5: ("#F3E8FD", "#8430CE"),
}
GRAY = "#5F6368"


def box(ax, cx, cy, w, h, title, body, idx, tsize=11.5, bsize=9.0):
    fc, ec = C[idx]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.018",
        linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy + h * 0.26, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color=ec, zorder=4)
    ax.text(cx, cy - h * 0.16, body, ha="center", va="center",
            fontsize=bsize, color="#202124", linespacing=1.55, zorder=4)


def route(ax, pts, color, lw=2.0, ls="-", arrow=True, rad=0.0):
    if arrow:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        if len(pts) > 2:
            ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=2)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=15, color=color,
                                    lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))
    else:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, ls=ls, zorder=2)


def label(ax, x, y, s, size=8.6, color=GRAY, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92),
            linespacing=1.4)


fig = plt.figure(figsize=(13.2, 10.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.16,
                      left=0.04, right=0.96, top=0.94, bottom=0.05)

# ============ 上图：交付关系 ============
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("五人协作与交付关系", fontsize=17, fontweight="bold", pad=10, color="#202124")
ax.text(0.5, 0.955, "实线 = 交付物（谁给谁什么）　虚线 = 复核关系",
        ha="center", fontsize=9.5, color=GRAY)

box(ax, 0.50, 0.805, 0.38, 0.150,
    "组员 1  组长 · 背景与总集成",
    "PPT 整合 · README · 排期与验收 · 提交包 · 演示视频\n讲 第 1—2 页（90 秒）", 1, tsize=12.5)

box(ax, 0.865, 0.545, 0.215, 0.170,
    "组员 2  原理讲解",
    "卷积/感受野/池化图\n原理讲稿 · 答疑卡\n讲 第 3—5 页（120 秒）", 2, tsize=11.5)

box(ax, 0.130, 0.215, 0.215, 0.180,
    "组员 3  数据与模型",
    "data.py · models.py\npredict.py · 配置\n讲 第 6—7 页（90 秒）", 3, tsize=11.5)

box(ax, 0.410, 0.215, 0.215, 0.180,
    "组员 4  训练与演示",
    "train.py · evaluate.py\n权重 · 日志 · 环境\n讲 第 8—10 页（150 秒）", 4, tsize=11.5)

box(ax, 0.690, 0.215, 0.215, 0.180,
    "组员 5  复评与可视化",
    "visualize.py · 结果表\n三类图 · 误判分析\n讲 第 11—14 页（150 秒）", 5, tsize=11.5)

# 数据模型 -> 训练
route(ax, [(0.2375, 0.215), (0.3025, 0.215)], C[3][1])
label(ax, 0.270, 0.327, "模块与接口（H2）", size=7.6)

# 训练 -> 复评
route(ax, [(0.5175, 0.215), (0.5825, 0.215)], C[4][1])
label(ax, 0.550, 0.327, "权重与日志（H4）", size=7.6)

# 复评 -> 组长
route(ax, [(0.690, 0.305), (0.690, 0.730)], C[5][1])
label(ax, 0.702, 0.370, "实验包与三类图（H5）", size=8.0, ha="left")

# 训练 -> 组长
route(ax, [(0.410, 0.305), (0.410, 0.730)], C[4][1])
label(ax, 0.446, 0.545, "代码页 · 演示脚本\nH5 / H6", size=8.2, ha="left")

# 数据模型 -> 组长（左侧绕行）
route(ax, [(0.130, 0.305), (0.130, 0.805), (0.310, 0.805)], C[3][1], lw=1.6)
label(ax, 0.160, 0.640, "第 6—7 页 · 配置", size=8.2, ha="left")

# 原理 -> 组长（右侧绕行）
route(ax, [(0.865, 0.630), (0.865, 0.865), (0.690, 0.865)], C[2][1], lw=1.6)
label(ax, 0.833, 0.905, "原理 3 页 · 讲稿", size=8.2, ha="right")

# 复评 -> 训练（独立复评，虚线）
route(ax, [(0.620, 0.125), (0.620, 0.088), (0.470, 0.088), (0.470, 0.125)],
      C[5][1], lw=1.6, ls=(0, (5, 3)))
label(ax, 0.545, 0.052, "⑤ 独立复评 ④ 的指标与公平性", size=7.8)

# 复核说明
ax.text(0.035, 0.052,
        "虚线复核：② 核对 ③ 的形状与参数量；② 核对 ⑤ 的特征图解释",
        fontsize=7.8, color=GRAY, va="center", ha="left")

# ============ 下图：汇报时间轴 ============
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
ax2.set_title("汇报顺序与时间分配（合计 600 秒 = 10 分钟上限）",
              fontsize=14, fontweight="bold", pad=6, color="#202124")

segs = [
    (0,   90,  1, "① 组员 1", "第 1—2 页", "1:30"),
    (90,  210, 2, "② 组员 2", "第 3—5 页", "2:00"),
    (210, 300, 3, "③ 组员 3", "第 6—7 页", "1:30"),
    (300, 450, 4, "④ 组员 4", "第 8—10 页（含演示）", "2:30"),
    (450, 600, 5, "⑤ 组员 5", "第 11—14 页", "2:30"),
]
Y, H = 0.52, 0.34
for s, e, idx, who, pages, dur in segs:
    x0, x1 = s / 600, e / 600
    fc, ec = C[idx]
    ax2.add_patch(Rectangle((x0, Y), x1 - x0, H, facecolor=fc, edgecolor=ec, linewidth=2, zorder=3))
    ax2.text((x0 + x1) / 2, Y + H * 0.64, who, ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=ec, zorder=4)
    ax2.text((x0 + x1) / 2, Y + H * 0.26, f"{pages}\n{dur}", ha="center", va="center",
             fontsize=8.2, color="#202124", linespacing=1.5, zorder=4)

for t, txt in [(0, "0:00"), (90, "1:30"), (210, "3:30"), (300, "5:00"), (450, "7:30"), (600, "10:00")]:
    x = t / 600
    ax2.plot([x, x], [Y - 0.05, Y], color=GRAY, lw=1)
    ax2.text(x, Y - 0.12, txt, ha="center", va="top", fontsize=8.5, color=GRAY)

ax2.annotate("", xy=(0.95, 0.30), xytext=(0.05, 0.30),
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=1, ls=(0, (2, 3))))
ax2.text(0.5, 0.24, "彩排目标：内容 9 分 30 秒以内（留 30 秒给切页与停顿）",
         ha="center", fontsize=9, color="#D93025")
ax2.text(0.5, 0.10, "汇报结束后另有 2 分钟答疑：按分工对应作答，① 任务与提交 ② 原理 ③ 数据与架构 ④ 训练与预测 ⑤ 公平性与结果",
         ha="center", fontsize=8.6, color=GRAY)

fig.savefig("docs/五人协作关系图.png", dpi=180, facecolor="white", bbox_inches="tight")
print("saved")
