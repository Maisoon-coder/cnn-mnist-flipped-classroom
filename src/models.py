"""两种模型：全连接基线（FC）与简易 CNN（组员 3 交付）。

结构完全按《CNN任务总体方案》§3.1、§3.2 实现，不自行增删层：

FC
    Flatten → Linear(784→256) → ReLU → Linear(256→10)          共 203,530 个参数

CNN（LeNet 风格）
    Conv1(1→6, 5×5, pad=2) → ReLU → MaxPool(2)                  → 6×14×14
    Conv2(6→16, 5×5, pad=0) → ReLU → MaxPool(2)                 → 16×5×5
    Flatten → 400 → Linear(400→120) → ReLU → Linear(120→84) → ReLU → Linear(84→10)
                                                                共 61,706 个参数

对组员 4、5 的接口承诺
----------------------
- 输入 ``[N, 1, 28, 28]``，输出 ``[N, 10]`` 的 **logits**（**不加 Softmax**，
  ``nn.CrossEntropyLoss`` 内部已含 log_softmax，再套一层等于算两遍）。
- ``CNNNet.forward_features(x)`` 返回四个命名位置的张量：
  ``conv1_relu`` ``[N,6,28,28]`` / ``pool1`` ``[N,6,14,14]`` /
  ``conv2_relu`` ``[N,16,10,10]`` / ``pool2`` ``[N,16,5,5]``。
  该方法与 ``forward`` 走同一批层、同一份权重，不存在"另一套网络"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

#: 每个名字对应的总参数量（含偏置），用于自检与 PPT 页核对
EXPECTED_PARAMS = {"fc": 203_530, "cnn": 61_706}


# ---------------------------------------------------------------- 全连接基线


class FCNet(nn.Module):
    """全连接基线：把 28×28 图像展平成 784 维向量后接两层线性变换。

    它存在的意义是"对照组"：同样的数据、同样的训练设置，
    唯一差别是它**没有**局部连接和空间权值共享。
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 784 = 28×28。这一层就吃掉 200,960 个参数，是"参数爆炸"的直观来源
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N,1,28,28] → [N,784]；展平保留像素值和顺序，只是不再告诉网络"谁挨着谁"
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))   # [N,256]
        return self.fc2(x)           # [N,10] logits


# ---------------------------------------------------------------- 简易 CNN


class CNNNet(nn.Module):
    """LeNet 风格的两层卷积 + 三层全连接。

    尺寸推演（单个样本）：
        1×28×28  --conv1(5×5,pad=2)--> 6×28×28  --pool--> 6×14×14
                 --conv2(5×5,pad=0)--> 16×10×10 --pool--> 16×5×5 → Flatten 400
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 卷积 1：padding=2 让 28×28 保持 28×28，边缘像素也能被完整覆盖到
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # 卷积 2：不再 padding，(14-5)/1+1 = 10，输出 16×10×10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # 池化层无可学习参数，两个池化共用同一个配置
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分类头：16×5×5 = 400，这个数字算错就会报 mat1 and mat2 shapes cannot be multiplied
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # -------------------------------------------------- 中间特征（给组员 5 作图用）

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回四个关键位置的张量，键名固定，修改前必须通知组员 5。

        命名已标明截取位置：``conv1_relu``/``conv2_relu`` 是"卷积后 + ReLU 后"，
        ``pool1``/``pool2`` 是池化后的结果，画图标注不要写错。
        """
        x = torch.relu(self.conv1(x))       # [N, 6, 28, 28]
        conv1_relu = x
        x = self.pool(x)                    # [N, 6, 14, 14]
        pool1 = x
        x = torch.relu(self.conv2(x))       # [N, 16, 10, 10]
        conv2_relu = x
        x = self.pool(x)                    # [N, 16, 5, 5]
        pool2 = x
        return {
            "conv1_relu": conv1_relu,
            "pool1": pool1,
            "conv2_relu": conv2_relu,
            "pool2": pool2,
        }

    # -------------------------------------------------- 前向

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 复用 forward_features，保证"训练用的前向"和"画图用的中间结果"
        # 走的是同一批层、同一份权重，不会出现两套实现数字对不上的情况
        feats = self.forward_features(x)
        x = torch.flatten(feats["pool2"], start_dim=1)   # [N,400]
        x = torch.relu(self.fc1(x))                      # [N,120]
        x = torch.relu(self.fc2(x))                      # [N,84]
        return self.fc3(x)                               # [N,10] logits


# ---------------------------------------------------------------- 统一入口

_MODEL_ALIASES = {"fc": "fc", "fcn": "fc", "mlp": "fc", "cnn": "cnn", "lenet": "cnn"}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """按名字创建模型，``name`` 取 ``"fc"`` 或 ``"cnn"``（大小写不敏感）。

    两模型独立初始化：每次调用都会新建实例，随机数由调用方的种子控制。
    """
    key = str(name).strip().lower()
    if key not in _MODEL_ALIASES:
        raise ValueError(f"未知模型名 {name!r}，只支持 'fc' 或 'cnn'")
    key = _MODEL_ALIASES[key]
    return FCNet(num_classes=num_classes) if key == "fc" else CNNNet(num_classes=num_classes)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计参数量（默认只统计需要梯度的参数，与"可训练参数量"口径一致）。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- 权重读写


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    model_name: str,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """保存权重，并把"这份权重是什么"一起写进去。

    只存 ``state_dict`` 会导致事后无法确认权重对应哪个模型、哪个种子、第几轮，
    验收时"权重能否追溯到配置和轮次"就会失败。因此统一存成字典：

        {"format_version": 1, "model": "cnn", "seed": 42, "epoch": 5,
         "model_state": ..., "extra": {...}}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": str(model_name).lower(),
        "seed": seed,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """读取权重文件，兼容"完整字典"和"裸 state_dict"两种存法。

    用 ``weights_only=False``：不同 torch 版本的默认值不一致，显式指定可以避免
    现场加载时报 ``UnpicklingError``（见《环境统一说明》§4）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到权重文件：{path}")
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:      # torch < 2.0 没有 weights_only 参数
        obj = torch.load(path, map_location=map_location)

    if not isinstance(obj, dict):
        raise ValueError(f"权重文件格式无法识别：{path}")
    if "model_state" in obj:
        return obj
    # 裸 state_dict：没有任何元信息，模型名需要调用方显式给出
    return {
        "format_version": 0,
        "model": None,
        "seed": None,
        "epoch": None,
        "model_state": obj,
        "extra": {},
    }


def load_model_from_checkpoint(
    path: str | Path,
    model_name: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, Dict[str, Any]]:
    """从权重文件还原一个**可直接推理**的模型（已切到 eval 模式）。

    参数
    ----
    model_name : 权重里没有记录模型名时（裸 state_dict）必须显式给出。

    返回 ``(model, meta)``，``meta`` 含 model / seed / epoch / extra 等字段，
    便于打印"这次预测用的是哪个权重"。
    """
    meta = load_checkpoint(path, map_location="cpu")
    name = model_name or meta.get("model")
    if not name:
        raise ValueError(
            "权重文件里没有模型名，请用 --model 指定是 fc 还是 cnn"
        )

    model = build_model(name)
    state = meta["model_state"]
    # 兼容 DataParallel 存出来的 "module." 前缀
    if any(str(k).startswith("module.") for k in state):
        state = {str(k)[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()          # 必须：影响 Dropout/BatchNorm 等层的行为（本模型没有，但习惯要立住）
    return model, meta
