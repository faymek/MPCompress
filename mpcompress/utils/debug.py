import hashlib
import torch
import torch.nn as nn


def extract_shapes(nested_structure):
    if isinstance(nested_structure, torch.Tensor):
        shape = tuple(nested_structure.shape)
        return f"tensor: {shape}"
    elif isinstance(nested_structure, bytes):
        return f"bytes: {len(nested_structure)}"
    elif isinstance(nested_structure, dict):
        return {k: extract_shapes(v) for k, v in nested_structure.items()}
    elif isinstance(nested_structure, list):
        return [extract_shapes(item) for item in nested_structure]
    elif isinstance(nested_structure, tuple):
        return [extract_shapes(item) for item in nested_structure]  # 递归处理列表
    else:
        return nested_structure  # 其他类型直接返回


def tensor_hash(tensor):
    """计算张量的哈希值"""
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def debug_sequential(
    sequential: nn.Sequential, x: torch.Tensor, name: str = "Sequential"
):
    """
    调试Sequential层，输出每个子层的权重哈希和输出哈希

    Args:
        sequential: 要调试的Sequential层
        x: 输入张量
        name: Sequential层的名称
    """
    print(f"\n=== 调试 {name} ===")
    print(f"输入形状: {x.shape}, 哈希: {tensor_hash(x)}")

    current = x
    for i, layer in enumerate(sequential):
        # 获取权重哈希
        weight_hash = (
            tensor_hash(layer.weight)
            if hasattr(layer, "weight") and layer.weight is not None
            else "No weight"
        )

        # 前向传播
        try:
            output = layer(current)
            print(f"  Layer {i}: {type(layer).__name__}")
            print(f"    输入哈希: {tensor_hash(current)}")
            print(f"    权重哈希: {weight_hash}")
            print(f"    输出哈希: {tensor_hash(output)}")
            print(f"    输出形状: {output.shape}")
            current = output
        except Exception as e:
            print(f"  Layer {i}: {type(layer).__name__} - 错误: {e}")
            break

    print(f"最终输出: {current.shape}, 哈希: {tensor_hash(current)}")
    print("=" * 50)
