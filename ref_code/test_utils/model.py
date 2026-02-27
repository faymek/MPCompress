import torch
from sandwich_model.sandwich_joint import CondTCM
from sandwich_model.processor import UNetGenerator
from sandwich_model.config import (
    TCM_CONFIG,
    TCM_HEAD_DIM,
    TCM_DROP_PATH_RATE,
    TCM_N,
    TCM_M
)


def load_model(model_path=None, device='cuda'):
    """加载增强模型"""
    net = CondTCM(config=TCM_CONFIG, head_dim=TCM_HEAD_DIM, 
              drop_path_rate=TCM_DROP_PATH_RATE, N=TCM_N, M=TCM_M)
    net = net.to(device)
    net.eval()
    
    # 加载模型权重
    if model_path is not None:
        dictory = {}
        checkpoint = torch.load(model_path, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            new_k = k.replace("module.", "")
            if 'tcm' in new_k:
                new_k = new_k.replace("tcm.", "")
            if 'detector' in new_k:
                continue
            dictory[new_k] = v
        msg = net.load_state_dict(dictory, strict=False)
        print(msg)
    return net

def load_processor(processor_path=None, device='cuda'):
    """加载处理器模型"""
    processor = UNetGenerator(in_channels=3, out_channels=3, features=64)
    if processor_path is None:
        processor_path = "/mnt/netdisk/车网_sandwich/traffic-network/ckp/UNet_generator/0408_LPIPS_model_epoch_100.pth"
    msg = processor.load_state_dict(torch.load(processor_path, map_location=device))
    print(msg)
    processor = processor.to(device)
    processor.eval()  # 设置为评估模式
    return processor


