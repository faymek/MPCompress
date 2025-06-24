import torch
import timm
import numpy as np

def test_rectangular_vit():
    # 设置不同的输入尺寸
    img_size = (256, 252)  # 矩形输入尺寸
    patch_size = 16
    
    # 创建模型
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        img_size=img_size,
        patch_size=patch_size,
        drop_path_rate=0.0,
    )
    model.eval()
    
    # 创建随机输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # 前向传播
    with torch.no_grad():
        output = model.forward_features(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证patch数量
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
    print(f"Patch数量: {num_patches}")
    
    return model, output

if __name__ == "__main__":
    model, output = test_rectangular_vit() 