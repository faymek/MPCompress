import torch

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