# CNN 原理与 MNIST 手写数字识别实战

> 翻转课堂 1 小组项目 · 10 分钟汇报 + 2 分钟答疑 · 5 人小组
> 组名：待填 ｜ 成员：待填

## 一、要回答的问题

**为什么 CNN 比普通全连接网络更适合利用图像的局部空间结构？这种设计在 MNIST 上体现出怎样的效果与参数量差异？**

本项目以 MNIST 手写数字识别为载体，完成三件事：

1. **原理讲透**——卷积如何计算、感受野如何增长、池化起什么作用；
2. **代码跑通**——PyTorch 实现「简单全连接网络」与「简易 CNN」，可训练、可评估、可预测；
3. **效果对比**——在同一数据集与同一训练配置下对比准确率与参数量。

不追求复杂网络与高阶技巧，重点是核心思想的准确表达。

## 二、对比的两个模型

| | 结构 | 参数量 |
|---|---|---:|
| **FC 基线** | `Flatten → Linear(784,256) → ReLU → Linear(256,10)` | 203,530 |
| **简易 CNN** | `Conv(1→6,5×5,p2) → ReLU → MaxPool(2) → Conv(6→16,5×5) → ReLU → MaxPool(2) → Flatten(400) → Linear(400,120) → Linear(120,84) → Linear(84,10)` | 61,706 |

CNN 参数比 FC 少 **141,824** 个，下降 **69.68%**（结构计算值，最终以程序统计为准）。

## 三、仓库结构

```text
.
├── README.md
├── requirements.txt          # 待补：实跑通过后锁定版本
├── config.json               # 待补：正式实验冻结配置
├── src/                      # 待补：data / models / train / evaluate / predict / visualize
├── docs/                     # 任务书、方案、分工、知识点
├── outputs/<run_id>/         # 待补：实验日志、权重、图表
├── demo/                     # 待补：示例图片与备用演示视频
└── slides/                   # 待补：最终 PPT
```

`data/` 目录不入库（MNIST 约 55 MB，见 `.gitignore`）。

## 四、环境

统一方案见 **[docs/环境统一说明.md](docs/环境统一说明.md)**，要点：

- 必须装 PyTorch 的是**组员 3、4、5**；组员 1 建议装；组员 2 不需要；
- Python 3.11—3.14 均可（推荐 3.12），装 **CPU 版** torch 即可；
- 三人版本必须一致，D4 首次跑通后由 `pip freeze` 锁定 `requirements.txt`。

装完运行自检（只读，不安装任何东西）：

```bash
python scripts/check_env.py
```

通过标准：最后一行输出 `RESULT: PASS`。

## 五、快速开始

待补。以下命令形如：

```bash
python -m src.train --model fc  --config config.json --run-id main_seed42_v1
python -m src.train --model cnn --config config.json --run-id main_seed42_v1
python -m src.evaluate --run-dir outputs/main_seed42_v1
python -m src.visualize --run-dir outputs/main_seed42_v1
python -m src.predict --checkpoint <CNN权重> --image <图片路径>
```

实现完成后须用**实际验证过的命令**替换本节，并注明执行目录。

## 六、主要结果

| 模型 | 测试准确率 | 测试 loss | 总参数量 |
|---|---|---:|---:|
| FC | 待实测 | 待实测 | 203,530 |
| CNN | 待实测 | 待实测 | 61,706 |

## 七、数据准备

### 7.1 获取 MNIST

首次运行会自动下载；若网络不可用，可手动将 4 个文件放入 `data/MNIST/raw/`（文件名不能改，保持 `.gz` 不解压）：

| 文件 | 大小 | MD5 |
|---|---:|---|
| `train-images-idx3-ubyte.gz` | 9,912,422 | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | 28,881 | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | 1,648,877 | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | 4,542 | `ec29112dd5afa0611ce80d1b7f02629c` |

镜像：`https://ossci-datasets.s3.amazonaws.com/mnist/`（`data/` 不入库）。

### 7.2 训练集抽样：30,000 张

为缩短训练时间，本项目**不使用全部 60,000 张训练样本**，而使用一个固定的分层子集：

| 项 | 取值 |
|---|---|
| 训练样本数 | **30,000**（每类 3,000，类别完全均衡） |
| 抽样方式 | 分层随机抽样，`numpy.random.default_rng(42)` |
| 索引文件 | `splits/train_30000_seed42.npy`（int64，升序，30,000 个不重复索引） |
| 测试样本数 | **10,000（全量，不抽样）** |

重新生成索引：

```bash
python scripts/make_subset.py --n 30000 --seed 42
```

使用方式（**两个模型必须读同一个索引文件**）：

```python
import numpy as np
import torch.utils.data as tud
from torchvision import datasets

idx = np.load("splits/train_30000_seed42.npy")
full = datasets.MNIST(root="data", train=True, download=False, transform=tf)
train_ds = tud.Subset(full, idx)
```

> ⚠️ **口径要求**：所有材料（PPT、README、答辩）中描述样本数时，必须写"训练集 30,000（分层抽样）/ 测试集 10,000（全量）"，**不得声称使用了全部 60,000 张**。FC 与 CNN 必须使用完全相同的索引文件，否则对比无效。

## 八、小组分工

见 [docs/CNN五人分工与协作细则.md](docs/CNN五人分工与协作细则.md)。

## 九、文档索引

| 文档 | 内容 |
|---|---|
| [任务书](docs/翻转课堂1-卷积神经网络（CNN）原理与手写数字识别实战.md) | 课程原始要求 |
| [总体方案](docs/CNN任务总体方案.md) | 数据、模型、训练、可视化、排期的实施规定 |
| [分工与协作细则](docs/CNN五人分工与协作细则.md) | 五人角色、任务编号、交接与验收 |
| [修订清单](docs/CNN修订清单.md) | 对两份方案的修订项与 17 天排期 |
| [必备知识点](docs/CNN必备知识点.md) | 小白向的原理速查与答辩问题 |

## 十、复现说明

- 随机种子固定为 `42`；
- 两模型使用相同数据划分、相同预处理、相同优化器与学习率；
- 正式结果取最后一轮权重，测试集不参与任何模型选择；
- 权重加载：`torch.load(path, map_location="cpu", weights_only=False)`。

## 十一、协作约定

- 一个文件只有一名写者，改动前先 `Pull`；
- 提交说明写清做了什么，例如「添加 models.py 的 CNN 定义」；
- 代码、文档进本仓库；PPT、视频、数据集走网盘与雨课堂。
