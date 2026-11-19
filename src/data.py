"""MNIST 数据的读取、预处理与数据加载器（组员 3 交付）。

设计原则
--------
1. 预处理只有一份实现（``build_transform``）。训练、评估、单图预测都调用它，
   这样不会出现"训练时用一套标准化参数、演示时用另一套"的经典事故。
2. 训练集打乱用的随机生成器是**独立**的（``make_generator``），不复用全局随机状态。
   原因：两个模型的可训练参数个数不同，初始化消耗的随机数个数也不同，如果打乱顺序
   依赖全局种子，FC 和 CNN 看到的样本顺序就会不一样，对比实验不公平。
3. 训练评估/测试加载器一律 **不打乱**，batch 顺序与 ``datasets.MNIST`` 的内部索引
   一一对应，组员 5 才能用 "第 i 个 batch 的第 j 个样本" 回查测试集原始索引。

对组员 4、5 的接口承诺
----------------------
- 输入张量：``[N, 1, 28, 28]`` float32，已用 mean=0.1307 / std=0.3081 标准化。
- 标签张量：``[N]`` int64，取值 0—9（可直接喂 ``nn.CrossEntropyLoss``）。
- ``get_loaders`` 返回三个加载器，三者共用同一份数据集对象，样本集合完全一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------- 常量

#: MNIST 官方统计量（在 0—1 缩放的像素上计算），全组统一使用，不得各改各的
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

#: 项目根目录 = src/ 的上一级。所有路径都基于它，不写死某台电脑的绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 数据根目录，torchvision 会在其下寻找 MNIST/raw/*.gz
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

#: 官方样本数，用于核对数据是否完整
EXPECTED_TRAIN_SIZE = 60000
EXPECTED_TEST_SIZE = 10000


# ---------------------------------------------------------------- 预处理


def build_transform() -> transforms.Compose:
    """返回训练/评估/预测共用的图像预处理。

    ``ToTensor()``：PIL 图像 (H,W) 或 (H,W,C) → float32 张量 [C,H,W]，并把 0—255 压到 0—1；
    ``Normalize``：逐通道做 ``(x - mean) / std``，把输入拉到近似零均值，收敛更快。

    注意：预测单图时**必须**复用这个函数，否则输入分布与训练时不一致，准确率会莫名下降。
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


# ---------------------------------------------------------------- 数据集


#: MNIST 四个原始文件（torchvision 实际读取的是**解压后**的无后缀文件）
_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def ensure_raw_extracted(root: str | Path) -> None:
    """把 ``data/MNIST/raw/*.gz`` 解压成 torchvision 认得的原始文件。

    为什么需要这一步：仓库里只放体积小的 ``.gz``，但 ``datasets.MNIST(download=False)``
    检查的是**解压后**的文件名（如 ``train-images-idx3-ubyte``）。只放压缩包会报
    ``Dataset not found``——这是个很容易被误判成"数据没下载对"的坑。

    已存在解压文件时直接返回，重复调用无副作用；文件缺失且无 .gz 时什么都不做，
    由 ``datasets.MNIST`` 给出标准报错。
    """
    import gzip
    import shutil

    raw_dir = Path(root) / "MNIST" / "raw"
    if not raw_dir.is_dir():
        return
    for name in _RAW_FILES:
        target = raw_dir / name
        if target.exists():
            continue
        gz = raw_dir / f"{name}.gz"
        if not gz.exists():
            continue
        # 先写临时文件再改名，避免中途被打断留下半个文件、下次又误判为"已解压"
        tmp = target.with_suffix(target.suffix + ".part")
        with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        tmp.replace(target)


def get_datasets(
    data_root: Optional[str | Path] = None,
    download: bool = False,
) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """加载 MNIST 训练集与测试集（图像已转成标准化张量）。

    参数
    ----
    data_root : 数据根目录，默认 ``<项目根>/data``；其下应有 ``MNIST/raw/*.gz``。
    download : 是否允许联网下载。课堂/离线环境保持 False——下载失败会让整个流程卡住，
               而且现场不应依赖网络。数据已随项目提供，正常不需要置 True。

    返回
    ----
    (train_dataset, test_dataset)，长度分别为 60000 / 10000。
    """
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    if not download:
        ensure_raw_extracted(root)
        missing = [name for name in _RAW_FILES if not (root / "MNIST" / "raw" / name).exists()]
        if missing:
            raise FileNotFoundError(
                "MNIST 数据不完整，缺少：" + "、".join(missing) + "\n"
                f"应放在：{root / 'MNIST' / 'raw'}\n"
                "需要四个文件（可只给 .gz，首次运行会自动解压）：\n"
                "  train-images-idx3-ubyte.gz / train-labels-idx1-ubyte.gz\n"
                "  t10k-images-idx3-ubyte.gz  / t10k-labels-idx1-ubyte.gz"
            )
    transform = build_transform()

    train_ds = datasets.MNIST(
        root=str(root), train=True, download=download, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(root), train=False, download=download, transform=transform
    )
    return train_ds, test_ds


# ---------------------------------------------------------------- 打乱种子


def make_generator(seed: int) -> torch.Generator:
    """创建只用于"数据打乱顺序"的随机生成器。

    与 ``torch.manual_seed`` 分开的原因是：模型初始化也会消耗随机数，
    若两条流程共用同一个全局状态，FC / CNN 的样本顺序就会不同，破坏控制变量。
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ---------------------------------------------------------------- 加载器

#: 对应 DataLoader 的 drop_last：必须为 False，即保留末尾不足一个 batch 的样本。
#: 60000 = 937×64 + 32、10000 = 156×64 + 16，丢掉短 batch 会让 n_samples 对不上，
#: 准确率口径就会与测试集不一致。
_DROP_LAST = False


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """唯一的 DataLoader 构造入口。

    所有加载器都从这里出来，是为了让 ``drop_last`` / ``shuffle`` / 打乱种子这类
    口径只在**一处**定义。否则改一处漏一处，两个模型的数据顺序就会悄悄分叉。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，收到 {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # 只有打乱时才需要生成器；不打乱时传了也不会生效，写清楚免得误读
        generator=make_generator(seed) if (shuffle and seed is not None) else None,
        num_workers=num_workers,
        drop_last=_DROP_LAST,
        pin_memory=pin_memory,          # GPU 训练时可开，CPU 训练开着没坏处但没收益
        persistent_workers=num_workers > 0,
    )


def get_train_loader(
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    data_root: Optional[str | Path] = None,
    download: bool = False,
) -> DataLoader:
    """训练用加载器：打乱顺序，末尾不足一个 batch 的样本保留。"""
    train_ds, _ = get_datasets(data_root=data_root, download=download)
    return _make_loader(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_train_eval_loader(
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    data_root: Optional[str | Path] = None,
    download: bool = False,
) -> DataLoader:
    """训练集"评估用"加载器：不打乱，用于每轮结束后统计训练集指标。"""
    train_ds, _ = get_datasets(data_root=data_root, download=download)
    return _make_loader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_test_loader(
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    data_root: Optional[str | Path] = None,
    download: bool = False,
) -> DataLoader:
    """测试用加载器：不打乱，且严禁参与任何参数更新或模型选择。"""
    _, test_ds = get_datasets(data_root=data_root, download=download)
    return _make_loader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_loaders(
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    data_root: Optional[str | Path] = None,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """一次性取得三类加载器，供 train.py 直接使用。

    返回 ``(train_loader, train_eval_loader, test_loader)``。
    三个加载器共享同一份数据集对象（只实例化一次），保证样本集合、
    预处理和索引顺序完全一致。
    """
    train_ds, test_ds = get_datasets(data_root=data_root, download=download)
    return (
        _make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=seed,
                     num_workers=num_workers, pin_memory=pin_memory),
        _make_loader(train_ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=pin_memory),
    )


def get_dataset_by_split(
    split: str,
    data_root: Optional[str | Path] = None,
    download: bool = False,
) -> datasets.MNIST:
    """按名字取原始数据集，用于"根据索引回查原图"（组员 5 的误判复查要用）。

    ``split`` 取 ``"train"`` 或 ``"test"``。返回的数据集带 transform，
    取 ``ds[i]`` 得到的是 ``(标准化张量, 标签)``；要拿 0—255 的原图请用
    ``get_raw_dataset_by_split``。
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split 只能是 'train' 或 'test'，收到 {split!r}")
    train_ds, test_ds = get_datasets(data_root=data_root, download=download)
    return train_ds if split == "train" else test_ds


def get_raw_dataset_by_split(
    split: str,
    data_root: Optional[str | Path] = None,
    download: bool = False,
) -> datasets.MNIST:
    """同上，但**不做标准化**，只做 ToTensor（像素 0—1）。

    画图、导出演示图片、核对"模型看错了什么"时用这个，
    避免把标准化后的负数像素直接当图像显示。
    """
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    if split not in {"train", "test"}:
        raise ValueError(f"split 只能是 'train' 或 'test'，收到 {split!r}")
    if not download:
        ensure_raw_extracted(root)
    return datasets.MNIST(
        root=str(root),
        train=(split == "train"),
        download=download,
        transform=transforms.ToTensor(),
    )
