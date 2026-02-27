import torch.nn as nn
from torch import Tensor
import torch
import math

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def closestDivisors(num):
    for x in range(int(num ** 0.5), 0, -1):
        y = num // x
        if x * y == num: return x, y

def quant_tensor(feat: torch.Tensor, 
          tile_blk_num: tuple, 
          feat_format: str = 'NHWC') -> tuple:
    """
    Quantize feature maps and tile them into a matrix with batch support.
    
    Args:
        feat: torch.Tensor
            - Shape: 
                - If 'NHWC': [N, H, W, C] (batch, height, width, channels)
                - If 'NCHW': [N, C, H, W] (batch, channels, height, width)
        tile_blk_num: tuple of (int, int)
            Number of tiles in (height, width) dimensions
        feat_format: str
            Input format ('NHWC' or 'NCHW')
    
    Returns:
        tuple: (tiled_matrices, meta_data)
            tiled_matrices: torch.Tensor (uint8)
                - Shape: [N, H*tile_blk_num[0], W*tile_blk_num[1]]
            meta_data: list of tuples
                Each tuple contains:
                - maxFeat: float
                - minFeat: float  
                - pad_size: tuple (int, int)
                - matrix_shape: tuple (int, int, int)
                - tile_blk_num: tuple (int, int)
    """
    quantBitDepth = 8
    maxQuantCoeff = 2**quantBitDepth - 1

    # Convert to NHWC format if needed
    if feat_format == 'NCHW':
        # [N, C, H, W] -> [N, H, W, C]
        feat = feat.permute(0, 2, 3, 1)  
    
    # Compute min/max per sample in batch
    # feat shape: [N, H, W, C]
    maxFeat = feat.amax(dim=(1, 2, 3), keepdim=True)  # [N, 1, 1, 1]
    minFeat = feat.amin(dim=(1, 2, 3), keepdim=True)  # [N, 1, 1, 1]
    featRange = maxFeat - minFeat                   # [N, 1, 1, 1]
    # print(minFeat)
    
    # Quantize features
    # [N, H, W, C] = ([N, H, W, C] - [N, 1, 1, C]) * scalar / [N, 1, 1, 1]
    dataSampleScaled = torch.round((feat - minFeat) * maxQuantCoeff / featRange)
    dataSampleScaled = dataSampleScaled.to(torch.uint8)  # [N, H, W, C]
    
    # Initialize tiled matrices
    N, H, W, C = feat.shape
    total_H = H * tile_blk_num[0]  # Total tiled height
    total_W = W * tile_blk_num[1]   # Total tiled width
    
    # Output tensor: [N, H*tile_blk_num[0], W*tile_blk_num[1]]
    tiled_matrices = torch.zeros((N, total_H, total_W), 
                                dtype=torch.uint8, 
                                device=feat.device)
    
    # Tile each sample in batch
    for i in range(tile_blk_num[0]):  # For each tile row
        for j in range(tile_blk_num[1]):  # For each tile column
            channel_idx = i * tile_blk_num[1] + j
            tiled_matrices[:, i*H:(i+1)*H, j*W:(j+1)*W] = dataSampleScaled[:, :, :, channel_idx]
    
    # Prepare per-sample metadata
    meta_data = []
    pad_size = (0, 0)  # No padding in current implementation
    
    for b in range(N):
        # Get per-channel max/min for this sample
        sample_max = maxFeat[b].squeeze()  # [1]
        sample_min = minFeat[b].squeeze()  # [1]
        
        # Use mean of channel-wise max/min as representative values
        mean_max = sample_max.float().item()
        mean_min = sample_min.float().item()
        
        meta_data.append((
            mean_max,                   # Representative max value
            mean_min,                   # Representative min value
            pad_size,                   # (pad_h, pad_w)
            (total_H, total_W, 1),      # Matrix shape (h, w, c)
            tile_blk_num                # (tile_h, tile_w)
        ))
    
    return tiled_matrices, meta_data


def dequant_tensor(tiled_matrices: torch.Tensor, 
                  meta_data: list, out_format: str = 'NCHW') -> torch.Tensor:
    """
    Reconstruct original tensor from tiled matrices and metadata.
    
    Args:
        tiled_matrices: torch.Tensor (uint8)
            - Shape: [N, H*tile_blk_num[0], W*tile_blk_num[1]]
        meta_data: list of tuples
            Each tuple contains:
            - maxFeat: float
            - minFeat: float  
            - pad_size: tuple (int, int)
            - matrix_shape: tuple (int, int, int)
            - tile_blk_num: tuple (int, int)
    
    Returns:
        torch.Tensor: Reconstructed feature tensor in NHWC format
            - Shape: [N, H, W, C]
    """
    quantBitDepth = 8
    maxQuantCoeff = 2**quantBitDepth - 1
    
    N = tiled_matrices.shape[0]
    total_H, total_W, _ = meta_data[0][3]  # Get shape from first sample's metadata
    tile_blk_num = meta_data[0][4]         # (tile_h, tile_w)
    
    # Calculate original H/W dimensions
    H = total_H // tile_blk_num[0]
    W = total_W // tile_blk_num[1]
    C = tile_blk_num[0] * tile_blk_num[1]  # Total channels
    
    # Initialize output tensor [N, H, W, C]
    reconstructed = torch.zeros((N, H, W, C), 
                              dtype=torch.float32,
                              device=tiled_matrices.device)
    
    for b in range(N):  # For each sample in batch
        # Get metadata for this sample
        maxFeat, minFeat, _, _, _ = meta_data[b]
        featRange = maxFeat - minFeat
        
        # Extract tiles and reconstruct channels
        for i in range(tile_blk_num[0]):  # For each tile row
            for j in range(tile_blk_num[1]):  # For each tile column
                channel_idx = i * tile_blk_num[1] + j
                # Extract the tile for this channel
                tile = tiled_matrices[b, i*H:(i+1)*H, j*W:(j+1)*W]
                # Dequantize and store in output tensor
                reconstructed[b, :, :, channel_idx] = (
                    tile.float() * featRange / maxQuantCoeff + minFeat
                )
    
    if out_format == 'NCHW':
        # Convert to NCHW format if needed
        reconstructed = reconstructed.permute(0, 3, 1, 2)
    
    return reconstructed

if __name__ == "__main__":
    # Example usage
    feat = torch.randn(16, 128, 64, 64)  # [N, C, H, W]
    n_channels = feat.shape[1]
    h, w = closestDivisors(n_channels)
    tile_hw = (h, w)
    tiled_matrices, meta_data = quant_tensor(feat, tile_hw, feat_format='NCHW')
    reconstructed = dequant_tensor(tiled_matrices, meta_data)
    
    print("Original Feature Shape:", feat.shape)
    print("Tiled Matrices Shape:", tiled_matrices.shape)
    print("Reconstructed Feature Shape:", reconstructed.shape)  # Should match original shape
    # print("Original Feature:\n", feat)
    # print("Reconstructed Feature:\n", reconstructed)
    # 判断重建结果误差总和是否在一定范围
    error = torch.abs(feat - reconstructed)
    error_ratio = error / (torch.abs(feat) + 1e-8)  # Avoid division by zero)
    error = error_ratio.mean().item()
    print(error)


def quant_tensor_no_tile(feat: torch.Tensor, 
          feat_format: str = 'NHWC') -> tuple:
    """
    Quantize feature maps.
    
    Args:
        feat: torch.Tensor
            - Shape: 
                - If 'NHWC': [N, H, W, C] (batch, height, width, channels)
                - If 'NCHW': [N, C, H, W] (batch, channels, height, width)
        tile_blk_num: tuple of (int, int)
            Number of tiles in (height, width) dimensions
        feat_format: str
            Input format ('NHWC' or 'NCHW')
    
    Returns:
        tuple: (tiled_matrices, meta_data)
            tiled_matrices: torch.Tensor (uint8)
                - Shape: [N, H*tile_blk_num[0], W*tile_blk_num[1]]
            meta_data: list of tuples
                Each tuple contains:
                - maxFeat: float
                - minFeat: float  
                - pad_size: tuple (int, int)
                - matrix_shape: tuple (int, int, int)
                - tile_blk_num: tuple (int, int)
    """
    quantBitDepth = 8
    maxQuantCoeff = 2**quantBitDepth - 1

    # Convert to NHWC format if needed
    if feat_format == 'NCHW':
        # [N, C, H, W] -> [N, H, W, C]
        feat = feat.permute(0, 2, 3, 1)  
    
    # Compute min/max per sample in batch
    # feat shape: [N, H, W, C]
    maxFeat = feat.amax(dim=(1, 2, 3), keepdim=True)  # [N, 1, 1, 1]
    minFeat = feat.amin(dim=(1, 2, 3), keepdim=True)  # [N, 1, 1, 1]
    featRange = maxFeat - minFeat                   # [N, 1, 1, 1]
    # print(minFeat)
    
    # Quantize features
    # [N, H, W, C] = ([N, H, W, C] - [N, 1, 1, 1]) * scalar / [N, 1, 1, 1]
    dataSampleScaled = torch.round((feat - minFeat) * maxQuantCoeff / featRange)
    dataSampleScaled = dataSampleScaled.to(torch.uint8)  # [N, H, W, C]
    
    # Initialize tiled matrices
    N, H, W, C = feat.shape

    # Prepare per-sample metadata
    meta_data = []
    pad_size = (0, 0)  # No padding in current implementation
    
    for b in range(N):
        # Get per-channel max/min for this sample
        sample_max = maxFeat[b].squeeze()  # [1]
        sample_min = minFeat[b].squeeze()  # [1]
        
        # Use mean of channel-wise max/min as representative values
        mean_max = sample_max.float().item()
        mean_min = sample_min.float().item()
        
        meta_data.append((
            mean_max,                   # Representative max value
            mean_min,                   # Representative min value
            pad_size,                   # (pad_h, pad_w)
            (H, W, C),      # Matrix shape (h, w, c)
        ))

    # # Convert back to NCHW format if needed
    # if feat_format == 'NCHW':
    #     dataSampleScaled = dataSampleScaled.permute(0, 3, 1, 2)  # [N, C, H, W]
    
    return dataSampleScaled, meta_data