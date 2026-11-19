# CNN 原理与 MNIST 手写数字识别实战

> 翻转课堂 1 小组项目 · 10 分钟汇报 + 2 分钟答疑 · 5 人小组
> 组名：待填 ｜ 成员：待填

## 一、要回答的问题

**为什么 CNN 比普通全连接网络更适合利用图像的局部空间结构？这种设计在 MNIST 上体现出怎样的效果与参数量差异？**

围绕这一个问题组织全部工作：原理讲解 + 可运行代码 + 同条件对比实验。

## 二、实测结果

同一数据集、同一随机种子、同一优化器与学习率，**唯一变量是网络结构**。

| 模型 | 结构 | 参数量 | 测试准确率 | 测试 loss | 误判 / 10000 |
|---|---|---:|---:|---:|---:|
| FC（全连接基线） | `Flatten → Linear(784,256) → ReLU → Linear(256,10)` | 203,530 | 97.19% | 0.0926 | 281 |
| **CNN** | `Conv(1,6,5,p2) → ReLU → MaxPool(2) → Conv(6,16,5) → ReLU → MaxPool(2) → Flatten(400) → Linear(400,120) → ReLU → Linear(120,84) → ReLU → Linear(84,10)` | **61,706** | **99.00%** | **0.0312** | **100** |

> **CNN 用少 69.68% 的参数，拿到高 1.81 个百分点的准确率。**

训练设置：`seed=42`、5 轮、batch 64、Adam `lr=1e-3`、无数据增强 / Dropout / BatchNorm / 学习率调度。
训练集 **60000 张全量**，测试集 **10000 张全量**。

逐轮测试准确率：

| epoch | FC | CNN |
|---:|---:|---:|
| 1 | 96.44% | 97.33% |
| 2 | 97.12% | 98.31% |
| 3 | 97.17% | 98.43% |
| 4 | 97.47% | 98.93% |
| 5 | 97.19% | **99.00%** |

> FC 第 4 轮比第 5 轮高，但**正式结果取最后一轮**，不按测试集挑选轮次。

每类准确率（差距最大的是数字 8）：

| 数字 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FC | 99.0 | 99.3 | 97.4 | 99.4 | 95.8 | 96.3 | 98.0 | 96.2 | **91.3** | 98.7 |
| CNN | 99.8 | 99.3 | 99.2 | 99.8 | 99.0 | 98.1 | 98.9 | 98.2 | **98.6** | 99.0 |

## 三、目录结构

```text
.
├── README.md
├── config.json                 正式实验配置（冻结）
├── src/
│   ├── data.py                 预处理 + 三类加载器 + 索引回查
│   ├── models.py               FC / CNN + forward_features + 权重读写
│   ├── train.py                通用训练入口
│   ├── evaluate.py             独立评估 + 误判清单导出
│   └── predict.py              单图预测
├── scripts/
│   ├── check_env.py            环境自检（只读）
│   ├── check_interfaces.py     模型/数据接口自检
│   ├── export_demo_images.py   从测试集导出演示图片
│   └── make_samples_grid.py    生成样本网格图
├── outputs/main_seed42_v1/     正式实验产物
│   ├── config.json  environment.txt
│   ├── metrics.csv  results.csv
│   ├── errors_fc.csv  errors_cnn.csv
│   └── checkpoints/{fc,cnn}_seed42_last.pt
├── demo/                       10 张演示图（0—9 各一张，来自测试集）
├── docs/                       任务书、方案、分工、交接文档
└── data/                       MNIST 原始文件（不入库）
```

## 四、环境

统一方案见 **[docs/环境统一说明.md](docs/环境统一说明.md)**。

- 必须装 PyTorch 的是**组员 3、4、5**；组员 1 建议装；组员 2 不需要
- Python 3.11—3.14 均可（本项目在 3.14.3 上验证），**CPU 版 torch 足够**
- 实测版本：`torch 2.15.0+cpu`、`torchvision 0.29.2`、`numpy 2.5.5`、`matplotlib 3.11.2`

装完先自检：

```bash
python scripts/check_env.py
```

通过标准：最后一行输出 `RESULT: PASS`。

## 五、快速开始

以下命令均在**项目根目录**执行，且都已实跑验证。

```bash
# 0) 环境自检
python scripts/check_env.py

# 1) 训练两个模型（同一 run-id 才算同一次对照实验）
python -m src.train --model fc  --config config.json --run-id main_seed42_v1
python -m src.train --model cnn --config config.json --run-id main_seed42_v1

# 2) 从权重独立重算指标，并导出误判清单
python -m src.evaluate --run-dir outputs/main_seed42_v1

# 3) 单图预测
python -m src.predict --checkpoint outputs/main_seed42_v1/checkpoints/cnn_seed42_last.pt \
                     --image demo/digit7_test0.png
```

**耗时参考**（CPU，8 线程）：FC 每轮约 35 秒、CNN 每轮约 45 秒，两个模型各 5 轮合计约 **10 分钟**。

## 六、数据准备

### 6.1 获取 MNIST

首次运行会自动下载；若网络不可用，可手动将 4 个文件放入 `data/MNIST/raw/`（文件名不能改，保持 `.gz` 不解压，首次运行会自动解压）：

| 文件 | 大小 | MD5 |
|---|---:|---|
| `train-images-idx3-ubyte.gz` | 9,912,422 | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | 28,881 | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | 1,648,877 | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | 4,542 | `ec29112dd5afa0611ce80d1b7f02629c` |

镜像：`https://ossci-datasets.s3.amazonaws.com/mnist/`（`data/` 不入库）

### 6.2 训练集用全量 60,000 张

项目早期讨论过"抽样 3 万 / 4 万"来加快训练，**实测后放弃**：全量 60000 训练 5 轮只要约 3 分钟，抽样省不下时间，反而掉 0.5—1 个百分点的准确率。

**所有材料中描述样本数时，必须写"训练 60,000 / 测试 10,000"。**

## 七、口径与限制

**必须守住的口径**

1. **取最后一轮**，不按测试集准确率挑选轮次
2. 准确率差用 **"百分点"**，参数量变化用 **相对比例**（减少 69.68%）
3. `errors_*.csv` 里的概率是模型打分，**不是置信度校准**

**已知限制**

- 只跑了 `seed=42` 一个种子，无法排除随机波动
- 基础版本不含数据增强 / Dropout / BatchNorm / 学习率调度（刻意设计，保证两模型只在结构上不同）
- CNN 第 5 轮仍在上升，加轮次还能涨，但按规定锁定 5 轮

## 八、文档索引

| 文档 | 内容 |
|---|---|
| [任务书](docs/翻转课堂1-卷积神经网络（CNN）原理与手写数字识别实战.md) | 课程原始要求 |
| [总体方案](docs/CNN任务总体方案.md) | 数据、模型、训练、可视化、排期的实施规定 |
| [分工与协作细则](docs/CNN五人分工与协作细则.md) | 五人角色、任务编号、交接与验收 |
| [修订清单](docs/CNN修订清单.md) | 对两份方案的修订项 |
| [必备知识点](docs/CNN必备知识点.md) | 小白向原理速查与答辩问题 |
| [环境统一说明](docs/环境统一说明.md) | 谁装什么、怎么装、怎么自检 |
| [五人协作关系图](docs/五人协作关系图.md) | 交付流向与复核关系 |
| [交接文档](docs/交接文档/) | 组员 4 的自留记录与致各成员的交接 |

## 九、协作约定

- 一个文件只有一名写者，改动前先 `git pull`
- 提交说明写清做了什么，例如「添加 models.py 的 CNN 定义」
- 代码、文档进本仓库；PPT、视频、数据集走网盘与雨课堂
