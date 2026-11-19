# 数据说明（组员 3 交付，README 第 6 页素材）

## 一、数据从哪来、放在哪

| 项目 | 内容 |
|---|---|
| 数据集 | MNIST 官方训练集 60,000 张 + 测试集 10,000 张 |
| 单图规格 | 28×28 灰度，像素 0—255，标签 0—9 |
| 项目内位置 | `data/MNIST/raw/` |
| 是否联网 | **不需要**。压缩包已随项目提供，`download=False` |
| 来源镜像 | `https://ossci-datasets.s3.amazonaws.com/mnist/` |

`data/MNIST/raw/` 里应同时存在 `.gz` 压缩包和 torchvision 读取的解压文件：

```text
data/MNIST/raw/
├── train-images-idx3-ubyte.gz      ← 随项目提供的原始压缩包（体积小）
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx1-ubyte.gz
├── train-images-idx3-ubyte         ← 首次运行时由 data.py 自动解压生成
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

> **踩过的坑**：`datasets.MNIST(download=False)` 检查的是**解压后**的文件名（不带 `.gz`）。
> 只放压缩包会报 `Dataset not found. You can use download=True to download it`，
> 很容易被误判成"数据没下载对"。所以 `src/data.py` 里加了 `ensure_raw_extracted()`：
> 第一次调用 `get_datasets()` 时自动解压，之后直接跳过。

## 二、预处理（全组唯一一份实现）

顺序：**读图 → 转灰度张量并缩放到 0—1 → 按 mean=0.1307、std=0.3081 标准化 → 组成 batch**

```python
transforms.Compose([
    transforms.ToTensor(),                                  # 0—255 → 0—1，HWC → CHW
    transforms.Normalize((0.1307,), (0.3081,)),             # (x - mean) / std
])
```

- 这两个常数是在 MNIST 训练集 0—1 像素上统计出来的官方值，全组统一使用。
- **预测时必须复用同一套参数**。`src/predict.py` 直接调用 `build_transform()`，
  不另写一份预处理——这是"演示时准确率莫名下降"的最常见原因。
- 训练 / 评估 / 预测三条链路共用此函数，改一处即三处生效。

## 三、三类数据加载器（给组员 4 的接口）

```python
from src.data import get_loaders

train_loader, train_eval_loader, test_loader = get_loaders(
    batch_size=64, seed=42, num_workers=0
)
```

| 加载器 | 是否打乱 | 用途 | 关键约束 |
|---|---|---|---|
| `train_loader` | ✅ 每轮重新打乱 | 参数更新 | 用**独立** `torch.Generator` 控制顺序 |
| `train_eval_loader` | ❌ | 每轮末统计训练集指标 | 与测试指标用同一个评估函数 |
| `test_loader` | ❌ | 报告正式结果、导出误判 | **绝不能**参与参数更新或挑模型 |

其它约定：

- `drop_last=False`：训练集 60000 = 937×64 + **32**，测试集 10000 = 156×64 + **16**，
  末尾短 batch 保留，否则 `n_samples` 对不上、准确率口径与测试集不一致。
- `num_workers=0` 起步：先排掉多进程带来的随机性与环境问题，跑通后再考虑加大。
- 标签是 `int64`（`torch.long`），可直接喂 `nn.CrossEntropyLoss`。

## 四、为什么打乱要用独立生成器

两个模型的可训练参数个数不同（203,530 vs 61,706），初始化消耗的随机数个数也不同。
如果数据打乱顺序依赖**同一个全局随机状态**，FC 和 CNN 每轮看到的样本顺序就会不一样，
"控制变量"就失效了。所以：

```python
generator = torch.Generator()
generator.manual_seed(seed)      # 只负责打乱顺序
DataLoader(..., shuffle=True, generator=generator)
```

模型初始化用 `torch.manual_seed(seed)` 单独控制，两条流程互不干扰。

## 五、索引与"回查原图"

测试加载器不打乱，因此 **batch 顺序 = `datasets.MNIST` 内部索引顺序**，
组员 5 可以按 `test_index` 直接取回原图：

```python
from src.data import get_raw_dataset_by_split

raw = get_raw_dataset_by_split("test")     # 只做 ToTensor，不标准化
image, label = raw[1234]                   # image: [1,28,28]，像素 0—1
```

- `get_dataset_by_split(split)`：带标准化，用于"模型输入长什么样"；
- `get_raw_dataset_by_split(split)`：不标准化，用于**画图显示**（避免把负数像素当图像）。
- 导出演示图片的脚本：`python scripts/export_demo_images.py`，输出到 `demo/`，
  并写出 `demo/来源说明.md` 记录每张图的测试集索引。

## 六、数据自检结论

`python scripts/check_interfaces.py` 实测输出（节选，只列与数据相关的项）：

```text
[PASS] batch 输入形状: (64, 1, 28, 28) dtype=torch.float32
[PASS] batch 标签类型: (64,) dtype=torch.int64 范围=[0,9]
[PASS] 训练集样本数: 60000
[PASS] 测试集样本数: 10000
[PASS] 训练评估加载器覆盖全部样本: 累计 60000 个样本
[PASS] 测试加载器覆盖全部样本: 累计 10000 个样本
[PASS] 末尾不足一个 batch 的样本被保留: 共 938 批，最后一批 32 张
[PASS] 同种子打乱顺序一致 / 不同种子打乱顺序不同
[PASS] 训练评估加载器不打乱 / 测试加载器不打乱
```

注意：这些都只证明"数据读取与批处理是对的"，**不涉及任何模型准确率**。
数据上没有跑过任何训练——训练与指标是组员 4 的 4-05，本部分不做。

## 七、第 6 页要讲的三句话

1. 60000 张训练、10000 张测试，**测试集不参与任何训练或调参**。
2. 预处理是 `0—1 缩放 + 标准化`，常数为 mean=0.1307、std=0.3081，两模型完全一致。
3. 训练打乱、测试不打乱；打乱用**独立种子**，保证 FC 与 CNN 看到完全相同的样本顺序。
