import torch
import os
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from test_utils.parser import parse_calcu_metric_args


def psnr_frames(args):
    # 设置设备
    device = 'cuda:0' if args.device == 'cuda' else 'cpu'

    # 设置文件夹路径
    folder1 = args.frame_dir
    folder2 = args.gt_dir

    # 创建存储结果的字典
    prefix_dict = defaultdict(list)
    all_distances = []

    # 遍历第一个文件夹
    for filename in tqdm(os.listdir(folder1)):
        # 构建对应文件的完整路径
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        
        # 确保第二个文件存在
        if not os.path.exists(file2_path):
            print(f"Warning: {filename} not found in target folder, skipping")
            continue

        try:
            # 加载并转换图像
            img1 = Image.open(file1_path).convert('RGB')
            img2 = Image.open(file2_path).convert('RGB')

            # 检查图像尺寸是否匹配
            if img1.size != img2.size:
                print(f"Warning: {filename} size mismatch ({img1.size} vs {img2.size}), skipping")
                continue

            # 转换为PyTorch Tensor
            img1_tensor = torch.tensor(np.array(img1)).to('cuda:0').permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2_tensor = torch.tensor(np.array(img2)).to('cuda:0').permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # 计算PSNR
            with torch.no_grad():
                mse = torch.mean((img1_tensor - img2_tensor) ** 2)
                psnr = -10.0 * torch.log10(mse)
            
            # 提取文件名前缀（假设前缀是第一个下划线前的部分）
            prefix = filename.split('_')[0]
            
            # 存储结果
            prefix_dict[prefix].append(psnr.item())
            all_distances.append(psnr.item())

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    # 计算并打印每个前缀的平均值
    print("\nPer-prefix average PSNR:")
    for prefix, distances in prefix_dict.items():
        avg = sum(distances) / len(distances)
        print(f"{prefix}: {avg:.4f}")

    # 计算并打印全局平均值
    if all_distances:
        global_avg = sum(all_distances) / len(all_distances)
        print(f"\nGlobal average PSNR: {global_avg:.4f}")
    else:
        print("\nNo valid image pairs found")


if __name__ == '__main__':
    args = parse_calcu_metric_args()
    psnr_frames(args)