"""接口与结构核对（组员 3 的交付自检，也是组员 2、4、5 的验收脚本）。

在项目根目录执行：

    python scripts/check_interfaces.py

逐项核对《CNN任务总体方案》§5.2 要求的检查，以及《分工细则》§4.3 的接口承诺：
输入形状、标签类型、两模型输出、参数量、中间特征、三类加载器、保存加载、单图预测。

全部输出 PASS 时最后一行打印 ``RESULT: PASS``。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms

from src.data import (
    EXPECTED_TEST_SIZE,
    MNIST_MEAN,
    MNIST_STD,
    EXPECTED_TRAIN_SIZE,
    get_datasets,
    get_loaders,
    get_raw_dataset_by_split,
    get_test_loader,
    get_train_loader,
)
from src.models import (
    EXPECTED_PARAMS,
    build_model,
    count_parameters,
    load_model_from_checkpoint,
    save_checkpoint,
)
from src.predict import load_image_as_array, predict_image, preprocess_image, predict_tensor

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    torch.manual_seed(42)

    # ---------------------------------------------------------- 1. 假输入前向
    dummy = torch.zeros(2, 1, 28, 28)
    for name in ("fc", "cnn"):
        model = build_model(name).eval()
        with torch.no_grad():
            out = model(dummy)
        check(f"{name} 假输入前向", tuple(out.shape) == (2, 10), f"[2,1,28,28] -> {tuple(out.shape)}")

    # ---------------------------------------------------------- 2. 真实 batch
    train_loader, train_eval_loader, test_loader = get_loaders(batch_size=64, seed=42)
    x, y = next(iter(train_loader))
    check("batch 输入形状", tuple(x.shape) == (64, 1, 28, 28),
          f"{tuple(x.shape)} dtype={x.dtype}")
    check("batch 标签类型", y.dtype == torch.int64 and int(y.min()) >= 0 and int(y.max()) <= 9,
          f"{tuple(y.shape)} dtype={y.dtype} 范围=[{int(y.min())},{int(y.max())}]")

    # 真实数据（不是全零假输入）也要能前向，且输出是有限数
    for name in ("fc", "cnn"):
        with torch.no_grad():
            out = build_model(name).eval()(x)
        check(f"{name} 真实 batch 前向",
              tuple(out.shape) == (64, 10) and bool(torch.isfinite(out).all()),
              f"输入 {tuple(x.shape)} -> 输出 {tuple(out.shape)}，取值有限")

    # ---------------------------------------------------------- 3. 参数量
    for name, expected in EXPECTED_PARAMS.items():
        got = count_parameters(build_model(name))
        check(f"{name} 参数量", got == expected, f"{got}（期望 {expected}）")

    # ---------------------------------------------------------- 4. 中间特征
    cnn = build_model("cnn").eval()
    expected_shapes = {
        "conv1_relu": (2, 6, 28, 28),
        "pool1": (2, 6, 14, 14),
        "conv2_relu": (2, 16, 10, 10),
        "pool2": (2, 16, 5, 5),
    }
    with torch.no_grad():
        feats = cnn.forward_features(dummy)
    check("特征键名", list(feats.keys()) == list(expected_shapes.keys()), str(list(feats.keys())))
    for key, expected_shape in expected_shapes.items():
        check(f"特征 {key} 形状", tuple(feats[key].shape) == expected_shape, str(tuple(feats[key].shape)))

    # forward_features 与 forward 必须走同一批权重：拼回去应能复现 forward 的 logits
    with torch.no_grad():
        manual = cnn.fc3(torch.relu(cnn.fc2(torch.relu(cnn.fc1(torch.flatten(feats["pool2"], 1))))))
        direct = cnn(dummy)
    check("特征与 forward 一致", torch.allclose(manual, direct, atol=1e-5),
          f"max|diff|={(manual - direct).abs().max().item():.3e}")

    # ---------------------------------------------------------- 5. 三类加载器
    train_ds, test_ds = get_datasets()
    check("训练集样本数", len(train_ds) == EXPECTED_TRAIN_SIZE, str(len(train_ds)))
    check("测试集样本数", len(test_ds) == EXPECTED_TEST_SIZE, str(len(test_ds)))

    n_train = sum(batch_y.numel() for _, batch_y in train_eval_loader)
    n_test = sum(batch_y.numel() for _, batch_y in test_loader)
    check("训练评估加载器覆盖全部样本", n_train == EXPECTED_TRAIN_SIZE, f"累计 {n_train} 个样本")
    check("测试加载器覆盖全部样本", n_test == EXPECTED_TEST_SIZE, f"累计 {n_test} 个样本")
    # drop_last=False 时最后一批应短于 batch_size，而不是被丢弃
    batches = list(get_train_loader(batch_size=64))
    last_batch = batches[-1][0].shape[0]
    check("末尾不足一个 batch 的样本被保留",
          len(batches) * 64 > EXPECTED_TRAIN_SIZE and last_batch == EXPECTED_TRAIN_SIZE % 64,
          f"共 {len(batches)} 批，最后一批 {last_batch} 张（60000 = 937×64 + {EXPECTED_TRAIN_SIZE % 64}）")

    # ---------------------------------------------------------- 6. 打乱种子独立性
    a = next(iter(get_train_loader(batch_size=8, seed=42)))[1]
    b = next(iter(get_train_loader(batch_size=8, seed=42)))[1]
    c = next(iter(get_train_loader(batch_size=8, seed=7)))[1]
    check("同种子打乱顺序一致", torch.equal(a, b), f"seed=42 首批标签 {a.tolist()}")
    check("不同种子打乱顺序不同", not torch.equal(a, c), f"seed=7 首批标签 {c.tolist()}")

    # ---------------------------------------------------------- 7. 评估加载器不打乱
    first_eval = next(iter(train_eval_loader))[1][0].item()
    first_test = next(iter(get_test_loader()))[1][0].item()
    check("训练评估加载器不打乱", first_eval == int(train_ds[0][1]),
          f"首批首样本标签={first_eval}, ds[0] 标签={int(train_ds[0][1])}")
    check("测试加载器不打乱", first_test == int(test_ds[0][1]),
          f"首批首样本标签={first_test}, ds[0] 标签={int(test_ds[0][1])}")

    # ---------------------------------------------------------- 8. 保存 / 加载往返
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "cnn_check.pt"
        save_checkpoint(build_model("cnn").eval(), ckpt, model_name="cnn", seed=42, epoch=5)
        model_a, meta = load_model_from_checkpoint(ckpt, device="cpu")
        model_b, _ = load_model_from_checkpoint(ckpt, device="cpu")
        with torch.no_grad():
            out_a, out_b = model_a(dummy), model_b(dummy)
        check("权重可重新加载", torch.equal(out_a, out_b) and meta.get("model") == "cnn",
              f"两次加载同输入输出一致, meta.model={meta.get('model')}, epoch={meta.get('epoch')}")

        # ------------------------------------------------------ 9. 单图预测链路
        raw = get_raw_dataset_by_split("test")
        ref_input = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))(raw[0][0])
        normalize_only = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

        mismatches = []
        max_diff = 0.0
        for index in range(10):                      # 连测 10 张，覆盖不同笔画
            image_tensor, label = raw[index]
            png_path = Path(tmp) / f"sample_{index}.png"
            Image.fromarray((image_tensor.squeeze(0) * 255).round().to(torch.uint8).numpy(),
                            mode="L").save(png_path)

            # 路径 A：数据集给出的张量，只补 Normalize（数据集已含 ToTensor）
            ref_tensor = normalize_only(image_tensor)
            tensor_pred, _ = predict_tensor(model_a, ref_tensor.unsqueeze(0))
            # 路径 B：PNG 落盘 → 读回 → 与训练同一套预处理 → 推理
            path_pred, _ = predict_image(ckpt, png_path, model_name="cnn")

            from_path = preprocess_image(load_image_as_array(png_path)).squeeze(0)
            max_diff = max(max_diff, (from_path - ref_tensor).abs().max().item())
            if tensor_pred != path_pred:
                mismatches.append((index, int(label), tensor_pred, path_pred))

        check("单图预测：张量路径与图片路径一致（10 张）",
              not mismatches,
              f"不一致 {mismatches}" if mismatches else "10 张全部一致（权重未训练，只比一致性）")
        check("图片往返预处理无偏移", max_diff < 1e-6,
              f"10 张图 max|diff|={max_diff:.3e}（PNG 8 位量化往返后应逐像素相同）")

        ref_pred, ref_probs = predict_tensor(model_a, ref_input.unsqueeze(0))
        check("概率分布归一", abs(float(ref_probs.sum()) - 1.0) < 1e-5,
              f"sum={float(ref_probs.sum()):.6f}")

    print("-" * 68)
    if failures:
        print(f"RESULT: FAIL（{len(failures)} 项未通过：{', '.join(failures)}）")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
