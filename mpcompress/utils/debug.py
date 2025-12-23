import hashlib
import torch
import torch.nn as nn
import numpy as np


def extract_shapes(nested_structure):
    """
    Extract shape information from nested data structures.

    Recursively processes nested structures (tensors, arrays, dicts, lists, tuples)
    and returns a structure with the same nesting but containing shape information
    instead of the actual data.

    Args:
        nested_structure (any): A nested structure that may contain torch.Tensor,
            np.ndarray, bytes, dict, list, tuple, or other types.

    Returns:
        nested_structure (any): A structure with the same nesting as the input.
    """
    if isinstance(nested_structure, torch.Tensor):
        shape = tuple(nested_structure.shape)
        return f"tensor: {shape}"
    elif isinstance(nested_structure, np.ndarray):
        shape = tuple(nested_structure.shape)
        return f"numpy: {shape}"
    elif isinstance(nested_structure, bytes):
        return f"bytes: {len(nested_structure)}"
    elif isinstance(nested_structure, dict):
        return {k: extract_shapes(v) for k, v in nested_structure.items()}
    elif isinstance(nested_structure, list):
        return [extract_shapes(item) for item in nested_structure]
    elif isinstance(nested_structure, tuple):
        return [
            extract_shapes(item) for item in nested_structure
        ]  # Recursively process tuple
    else:
        return nested_structure  # Return other types as-is


def tensor_hash(x):
    """
    Compute SHA256 hash of a tensor or numpy array.

    Args:
        x (torch.Tensor or np.ndarray): A torch.Tensor or np.ndarray to compute hash for.

    Returns:
        hash (str): A hexadecimal string representing the SHA256 hash of the tensor/array data.

    Raises:
        ValueError: If the input type is not torch.Tensor or np.ndarray.
    """
    if isinstance(x, torch.Tensor):
        return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    elif isinstance(x, np.ndarray):
        return hashlib.sha256(x.tobytes()).hexdigest()
    else:
        raise ValueError(f"Unsupported type: {type(x)}")


def debug_sequential(
    sequential: nn.Sequential, x: torch.Tensor, name: str = "Sequential"
):
    """
    Debug a Sequential layer by printing weight hashes and output hashes for each sub-layer.

    This function processes the input through each layer in the Sequential module,
    printing detailed information about inputs, weights, and outputs at each step.
    Useful for debugging and verifying layer-by-layer transformations.

    Args:
        sequential (nn.Sequential): The Sequential layer to debug.
        x (torch.Tensor): Input tensor to pass through the Sequential layer.
        name: Name of the Sequential layer for identification in output.
    """
    print(f"\n=== Debugging {name} ===")
    print(f"Input shape: {x.shape}, Hash: {tensor_hash(x)}")

    current = x
    for i, layer in enumerate(sequential):
        # Get weight hash
        weight_hash = (
            tensor_hash(layer.weight)
            if hasattr(layer, "weight") and layer.weight is not None
            else "No weight"
        )

        # Forward pass
        try:
            output = layer(current)
            print(f"  Layer {i}: {type(layer).__name__}")
            print(f"    Input hash: {tensor_hash(current)}")
            print(f"    Weight hash: {weight_hash}")
            print(f"    Output hash: {tensor_hash(output)}")
            print(f"    Output shape: {output.shape}")
            current = output
        except Exception as e:
            print(f"  Layer {i}: {type(layer).__name__} - Error: {e}")
            break

    print(f"Final output: {current.shape}, Hash: {tensor_hash(current)}")
    print("=" * 50)
