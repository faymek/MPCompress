import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_slide_windows(img_size, crop_size, stride):
    """可视化滑动窗口的位置"""
    h_img, w_img = img_size
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    
    # 计算网格数量
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    # 创建白色背景
    plt.imshow(np.ones((h_img, w_img)), cmap='gray', vmin=0, vmax=1)
    
    # 生成不同的颜色
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # 绘制每个窗口
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            
            # 选择颜色
            color_idx = (h_idx * w_grids + w_idx) % len(colors)
            color = colors[color_idx]
            
            # 绘制矩形（右边和下边减少1px）
            rect = plt.Rectangle((x1, y1), x2-x1-1, y2-y1-1, 
                               fill=False, edgecolor=color, linewidth=1)
            plt.gca().add_patch(rect)
            
            # 添加坐标标签（使用黑色文字）
            plt.text(x1, y1, f'({x1},{y1})', color='black', fontsize=8)
            plt.text(x2-1, y2-1, f'({x2},{y2})', color='black', fontsize=8)
    
    plt.title(f'Sliding Windows (Image: {h_img}x{w_img}, Crop: {h_crop}x{w_crop}, Stride: {h_stride}x{w_stride})')
    plt.axis('off')
    plt.savefig('slide_windows_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_slide_inference():
    # 设置参数
    img_size = (512, 817)  # 图像尺寸
    crop_size = (512, 512)  # 裁剪尺寸
    stride = (256, 256)    # 步长
    
    # 创建模拟图像
    img = torch.randn(1, 3, img_size[0], img_size[1])
    
    # 计算网格数量
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    h_img, w_img = img_size
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    
    print(f"图像尺寸: {img_size}")
    print(f"裁剪尺寸: {crop_size}")
    print(f"步长: {stride}")
    print(f"网格数量: {h_grids}x{w_grids}")
    print("\n滑动窗口坐标:")
    
    # 模拟推理过程
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            
            print(f"窗口 {h_idx*w_grids + w_idx + 1}:")
            print(f"  左上角: ({x1}, {y1})")
            print(f"  右下角: ({x2}, {y2})")
            print(f"  窗口大小: {x2-x1}x{y2-y1}")
            print()
    
    # 可视化滑动窗口
    visualize_slide_windows(img_size, crop_size, stride)

if __name__ == "__main__":
    test_slide_inference() 