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

### 数字 8：差距最大的一类（写第 14 页用这组数）

测试集里 8 共 **974** 张（不是 1000）。

| 项 | FC | CNN |
|---|---:|---:|
| 该类误判数 / 974 | **85** | **14** |
| 该类准确率 | 91.27% | 98.56% |
| 主要误判去向 | 8→9（26）、8→3（25）、8→5（13）、8→4（7）、8→7（5）、8→0（5）、8→2（2）、8→6（2） | 8→3（5）、8→9（3）、8→6（2）、8→7/0/4/2 各 1 |

两模型误判集合的关系（可佐证"差别来自结构"）：

| 统计 | 数量 | 自洽性 |
|---|---:|---|
| FC 误判总数 | 281 | = 220（仅 FC 错）+ 61（两者都错） |
| CNN 误判总数 | 100 | = 61（两者都错）+ 39（仅 CNN 错） |
| 仅 FC 错、CNN 对 | **220** | — |
| 其中真实为 8 的 | **74** | 85（FC 的 8 误判）= 74 + 11（两者都错） |

> 计数口径：按 `errors_*.csv` 中 `true_label==8` 统计；**一切以本表与 [`analysis/error_summary.md`](analysis/error_summary.md) 为准。**

### 训练耗时

`train_seconds` 记录的是**每轮**耗时；5 轮累计 **FC 185.0 s / CNN 220.8 s**（CPU，torch 默认线程数 8，未显式设置）。

> 参数更少不等于算得更快：CNN 前向的卷积运算比全连接矩阵乘更重，所以反而更慢。这一点建议主动讲。

### 复现性：三台不同环境，结果一致

| 谁 | 环境 | 结果 |
|---|---|---|
| 组员 4（本仓库产出） | Python 3.14.3 / torch 2.15.0+cpu / CPU | 99.00% / 97.19% |
| 组员 3（交叉核对） | Python 3.12.15 / torch 2.11.0+cu128 / RTX 5070 | 完全相同 |
| 组员 5（独立复评） | Python 3.13.14 / torch 2.15.0+cpu | 完全相同，误判逐条 diff 无差异 |

### 图表与数据出处

| 产物 | 位置 | 生成方式 |
|---|---|---|
| 9 张图（学习曲线、卷积核、特征图、误判网格、同图对照等） | [`figures/`](figures/) | `python visualize.py` 由 `outputs/` 重新生成，无手填数字 |
| 第 8—10 页三张图（结构图、训练循环、单图预测） | [`figures/fig1[012]_*.png`](figures/) | `python scripts/make_ppt_figs.py`，结构取自实跑统计，概率来自真实前向 |
| 结果表、每类准确率、误判配对、公平性检查表 | [`analysis/`](analysis/) | 脚本生成 |
| 第 11—14 页文案（含 150 秒口播） | [docs/组员5/PPT_11-14文案.md](docs/组员5/PPT_11-14文案.md) | — |

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
├── slides/                     PPT 分页文件（组员 3：第 6—7 页）
├── figures/                    9 张结果图（组员 5，由 visualize.py 生成）
├── analysis/                   结果表、每类准确率、误判配对、公平性检查表
├── visualize.py                统一绘图入口，一次生成全部图表
├── docs/                       任务书、方案、分工、交接文档
│   ├── 组员4/                  第 8—10 页文案、答疑卡
│   └── 组员5/                  复评报告、PPT 11—14 文案、答疑卡、验收自检表
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

## 八、接口约定（组员 4、5 直接调用）

| 内容 | 约定 |
|---|---|
| 输入张量 | `[N, 1, 28, 28]` float32，已标准化（mean=0.1307，std=0.3081） |
| 标签张量 | `[N]` int64，取值 0—9 |
| 模型输出 | `[N, 10]` **logits，不含 Softmax** |
| 模型名 | `"fc"` / `"cnn"`，统一入口 `build_model(name)` |
| 参数量 | FC 203,530；CNN 61,706（均含偏置） |
| CNN 特征 | `forward_features(x)` → `conv1_relu` `[N,6,28,28]`、`pool1` `[N,6,14,14]`、`conv2_relu` `[N,16,10,10]`、`pool2` `[N,16,5,5]` |
| Flatten 大小 | 400 |
| 加载器 | `get_loaders(batch_size, seed, num_workers)` → `(train, train_eval, test)` |
| 权重格式 | `save_checkpoint(...)` 写 dict；`load_model_from_checkpoint(path)` 还原出已 `eval()` 的模型 |
| 索引回查 | `get_raw_dataset_by_split("test")[i]` 拿 0—1 像素原图 |

训练循环见 [`src/train.py`](src/train.py)，评估与误判导出见 [`src/evaluate.py`](src/evaluate.py)。

```python
from src.data import get_loaders
from src.models import build_model, count_parameters

train_loader, train_eval_loader, test_loader = get_loaders(batch_size=64, seed=42)
model = build_model("cnn")
print(count_parameters(model))          # 61706
# zero_grad → forward → CrossEntropy(logits, y) → backward → step
```

> 三条最容易踩的坑：**输出不要套 Softmax**、**测试集不打乱才能按索引回查**、**预测必须复用 `build_transform()`**。

## 九、常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError: MNIST 数据不完整` | `data/MNIST/raw/` 里没有四个文件 | 按报错列出的四个文件名补上，层级必须是 `data/MNIST/raw/` |
| `mat1 and mat2 shapes cannot be multiplied` | Flatten 后维度不是 400 | 检查 Conv1 的 padding 是否为 2、两个池化是否都是 2×2 |
| `ModuleNotFoundError: No module named 'src'` | 没在项目根目录执行，或没加 `-m` | 回到项目根目录，用 `python -m src.predict ...` |
| `FileNotFoundError: 找不到权重文件` | 相对路径起点不是项目根 | 用绝对路径，或先 `cd` 到项目根 |
| 预测结果明显离谱 | 图片极性反了 / 不是 28×28 的 MNIST 风格 | 白底黑字加 `--invert`；确认数字居中、未被裁切 |
| torch 加载权重报 `UnpicklingError` | torch 版本不一致 | `load_checkpoint` 已固定 `weights_only=False`，仍失败则统一版本 |

## 十、文档索引

| 文档 | 内容 |
|---|---|
| [任务书](docs/翻转课堂1-卷积神经网络（CNN）原理与手写数字识别实战.md) | 课程原始要求 |
| [总体方案](docs/CNN任务总体方案.md) | 数据、模型、训练、可视化、排期的实施规定 |
| [分工与协作细则](docs/CNN五人分工与协作细则.md) | 五人角色、任务编号、交接与验收 |
| [必备知识点](docs/CNN必备知识点.md) | 原理速查与答辩问题 |
| [环境统一说明](docs/环境统一说明.md) | 谁装什么、怎么装、怎么自检 |
| [交接文档](docs/交接文档/) | 组员 4 的自留记录与致各成员的交接 |
| [组员3任务完成说明](docs/组员3任务完成说明.md) | 组员 3 的九项任务完成情况与 H2 / H2b 验收记录 |
| [H4 复评报告](docs/组员5/复评报告_H4.md) | 组员 5 的独立复评过程与结论 |
| [第 11—14 页文案](docs/组员5/PPT_11-14文案.md) | 组员 5：含 150 秒口播与预判追问 |
| [答疑卡](docs/组员5/答疑卡.md) | 八问 + 备用数字 |
| [验收自检表](docs/组员5/验收自检表.md) | 5-01 ~ 5-08 对照细则逐条自检 |
| [第 8—10 页文案](docs/组员4/PPT_8-10文案.md) | 组员 4：结构/训练循环/单图预测，含 150 秒口播 |
| [答疑卡（组员4）](docs/组员4/答疑卡.md) | 必答六问 + 备用数字 |
| [第 6—7 页 PPT](slides/组员3_第6-7页.pptx) | 组员 3：数据预处理流水线、网络结构（90 秒） |

## 十一、协作约定

- 一个文件只有一名写者，改动前先 `git pull`
- 提交说明写清做了什么，例如「添加 models.py 的 CNN 定义」
- 代码、文档进本仓库；PPT、视频、数据集走网盘与雨课堂
