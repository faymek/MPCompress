import torch
import torch.nn as nn


def extract_patches(image, patch_size=512, stride=256):
    """从图像中提取重叠的裁剪，并返回填充后的高度和宽度。"""
    _, H, W = image.shape
    patches = []
    positions = []
    
    # 计算需要填充的高度和宽度
    pad_h = max(patch_size - H, 0)
    pad_w = max(patch_size - W, 0)
    
    if pad_h > 0 or pad_w > 0:
        pad_layer = nn.ReplicationPad2d((0, pad_w, 0, pad_h))
        image = pad_layer(image.unsqueeze(0)).squeeze(0)
        H, W = image.shape[1], image.shape[2]
    
    # 提取裁剪
    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            patch = image[:, top:top + patch_size, left:left + patch_size]
            patches.append(patch)
            positions.append((top, left))
    
    # 处理底部不足patch_size的区域
    if (H - patch_size) % stride != 0:
        top = H - patch_size
        for left in range(0, W - patch_size + 1, stride):
            patch = image[:, top:top + patch_size, left:left + patch_size]
            if (top, left) not in positions:
                patches.append(patch)
                positions.append((top, left))
    
    # 处理右侧不足patch_size的区域
    if (W - patch_size) % stride != 0:
        left = W - patch_size
        for top in range(0, H - patch_size + 1, stride):
            patch = image[:, top:top + patch_size, left:left + patch_size]
            if (top, left) not in positions:
                patches.append(patch)
                positions.append((top, left))
    
    # 处理右下角不足patch_size的区域
    if ((H - patch_size, W - patch_size) not in positions):
        top = H - patch_size
        left = W - patch_size
        patch = image[:, top:top + patch_size, left:left + patch_size]
        patches.append(patch)
        positions.append((top, left))
    
    return patches, positions, H, W


def reconstruct_image(patches, positions, image_size, patch_size=512, stride=256):
    """将增强后的裁剪拼接回原始图像，重叠区域进行加权平均。"""
    C, H, W = image_size
    reconstructed = torch.zeros((C, H, W), dtype=patches[0].dtype)
    weight = torch.zeros((C, H, W), dtype=patches[0].dtype)
    
    for patch, (top, left) in zip(patches, positions):
        end_top = min(top + patch_size, H)
        end_left = min(left + patch_size, W)
        current_patch_size_h = end_top - top
        current_patch_size_w = end_left - left
        
        if current_patch_size_h != patch_size or current_patch_size_w != patch_size:
            patch = patch[:, :current_patch_size_h, :current_patch_size_w]
        
        reconstructed[:, top:end_top, left:end_left] += patch
        weight[:, top:end_top, left:end_left] += 1.0
    
    weight[weight == 0] = 1.0
    reconstructed /= weight
    return reconstructed