import cv2
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import subprocess
import os
import pickle
import json

from .config import (
    IMG_FORMAT,
    THREADS
)
from sandwich_model.feature_extractor import (
    setup_unet_feature_extractor,
    extract_unet_features
)
from sandwich_model.utils import (
    closestDivisors,
    quant_tensor,
    dequant_tensor
)

def save_yuv_single_frame(quant_data, save_dir):
    fp = open(save_dir, 'wb')
    strs = np.squeeze(quant_data).tobytes()
    fp.write(strs)
    fp.close()

def load_video(video_path):
    """返回格式：(参数元组, 帧生成器)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 定义帧生成器
    def frame_generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
    
    return (fps, (width, height), total_frames), frame_generator()


def process_video(model, input_path, output_path, device, mod=None, output_feature_path=None):
    """修正后的处理函数（输入输出均为[0,1]范围）"""
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
    if output_feature_path:
        os.makedirs(output_feature_path, exist_ok=True)
    
    # 定义转换（仅将numpy数组转tensor，不改变数值范围）
    transform = transforms.Compose([
        transforms.ToTensor()  # [0,255] -> [0,1] 并转换为CHW
    ])
    
    model.eval()
    with torch.no_grad():
        for idx, frame in enumerate(tqdm(frame_gen, total=total_frames, desc="Processing")):
            # 步骤1: BGR转RGB并转换为[0,1]范围的tensor
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_frame).unsqueeze(0).to(device)  # [1,3,H,W]
            # 步骤2: 模型处理（假设输出已在[0,1]范围）
            if mod == 0:
                enhanced_tensor = input_tensor
            else:
                enhanced_tensor = model.preprocess(input_tensor, mod)

            # 步骤4: 转换为可写入视频的格式
            output_frame = enhanced_tensor.squeeze().cpu().numpy()  # [3,H,W]
            output_frame = np.transpose(output_frame, (1, 2, 0))   # 转HWC
            output_frame = np.clip(output_frame,0.0,1.0)
            output_frame = (output_frame*255).astype(np.uint8)    # [0,1] -> [0,255]
            cv2.imwrite("output.png",cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

            residual_frame = (enhanced_tensor - input_tensor).squeeze().cpu().numpy()
            residual_frame = np.transpose(residual_frame, (1, 2, 0))   # 转HWC
            residual_frame = np.clip(residual_frame,0.0,1.0)
            residual_frame = (residual_frame*255).astype(np.uint8)    # [0,1] -> [0,255]
            # print((residual_frame == 0).mean())
            cv2.imwrite("residual.png",cv2.cvtColor(residual_frame, cv2.COLOR_RGB2BGR))
            
            # 步骤5: 写入视频
            writer.write(output_frame)
    
    writer.release()
    return fps, frame_size, total_frames


def get_frame_count_opencv(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("无法打开视频文件")
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def extract_video_frame(video_path, interval, ffmpeg_path, output_dir) -> bool:
    """对单个视频执行抽帧"""
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_tpl = os.path.join(output_dir, f"{video_name}_%08d.{IMG_FORMAT}")
        # os.makedirs(os.path.join(output_dir, video_name), exist_ok=True)

        # ffmpeg 命令
        cmd = [
            ffmpeg_path,
            "-i", video_path,          # 输入
            "-vsync", "0",             # 保留原始时间戳，避免丢帧/重复
            "-compression_level", "0", # png 无损；若用 jpg 可换成 -qscale:v 1
            "-threads", str(THREADS),
            "-vf", f"select=not(mod(n\,{interval}))",  # 每隔interval帧抽一帧
            "-y",                      # 覆盖已有文件
            output_tpl
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"[抽帧失败] {os.path.basename(video_path)}\n{e.output.decode()}")
        return False
    except Exception as e:
        print(f"[未知错误] {os.path.basename(video_path)}\n{e}")
        return False