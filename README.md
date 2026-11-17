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

待补。待代码实跑通过后由 `requirements.txt` 锁定精确版本（Python / PyTorch / torchvision / NumPy），并在 `environment.txt` 记录设备型号与是否启用 CUDA。

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

首次运行会自动下载 MNIST；若网络不可用，可手动将 4 个 `idx.gz` 文件放入 `data/MNIST/raw/`。

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
