import torch
from torchvision.models import vgg16
import torchvision.transforms as transforms
from torch.nn.functional import normalize
import cv2
from tqdm import tqdm
import numpy as np
import os
from .video import load_video

def setup_vgg_feature_extractor(device='cuda'):
    """初始化VGG16特征提取器（使用conv4_3层输出）"""
    model = vgg16(pretrained=True).features[:23]  # 取到conv4_3层
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def extract_vgg_features(vgg_model, frame_tensor, normalize_features=True):
    """
    提取VGG特征（与process_video并行执行）
    参数:
        vgg_model: 预初始化的VGG特征提取器
        frame_tensor: 输入图像张量 [1,3,H,W], 范围[0,1]
        normalize_features: 是否对特征做L2归一化
    返回:
        features: 提取的特征张量 [1,C,H',W']
    """
    # VGG预处理: [0,1] -> ImageNet标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(frame_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(frame_tensor.device)
    normalized_frame = (frame_tensor - mean) / std
    
    # 提取特征
    with torch.no_grad():
        features = vgg_model(normalized_frame)
    
    if normalize_features:
        features = normalize(features, p=2, dim=1)  # 沿通道维度L2归一化
    
    return features

# 修改后的process_video函数（添加特征提取）
def process_video_with_features(model, input_path, output_path, device, mod=None, feature_dir=None):
    """同时处理视频和提取VGG特征"""
    # 初始化VGG特征提取器
    vgg_extractor = setup_vgg_feature_extractor(device)
    
    # 获取参数和生成器
    (fps, frame_size, total_frames), frame_gen = load_video(input_path)
    
    # 初始化写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_path, 
        fourcc, 
        fps, 
        (frame_size[0], frame_size[1])
    )
    
    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor()  # [0,255] -> [0,1] 并转换为CHW
    ])
    
    model.eval()
    with torch.no_grad():
        for idx, frame in enumerate(tqdm(frame_gen, total=total_frames, desc="Processing")):
            # 步骤1: 帧预处理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_frame).unsqueeze(0).to(device)  # [1,3,H,W]
            
            # 步骤2: 并行执行
            enhanced_tensor = model.preprocess(input_tensor, mod)  # 原处理流程
            features = extract_vgg_features(vgg_extractor, input_tensor)  # 特征提取
            
            # 可选: 保存特征
            if feature_dir:
                os.makedirs(feature_dir, exist_ok=True)
                torch.save(features.cpu(), os.path.join(feature_dir, f"frame_{idx:05d}.pt"))
            
            # 步骤3: 视频写入（保持原流程不变）
            output_frame = enhanced_tensor.squeeze().cpu().numpy()  # [3,H,W]
            output_frame = np.transpose(output_frame, (1, 2, 0))   # 转HWC
            output_frame = np.clip(output_frame,0.0,1.0)
            output_frame = (output_frame*255).astype(np.uint8)    # [0,1] -> [0,255]
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            writer.write(output_frame)
    
    writer.release()
    return fps, frame_size, total_frames