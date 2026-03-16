import torch
from torchvision.models import vgg16
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
    
def setup_vgg_feature_extractor(device='cuda'):
    """初始化VGG16特征提取器"""
    model = vgg16(pretrained=True).features
    model = model.to(device)
    return model

def extract_vgg_features(vgg_model, frame_tensor):
    """
    提取VGG特征（与process_video并行执行）
    参数:
        vgg_model: 预初始化的VGG特征提取器
        frame_tensor: 输入图像张量 [1,3,H,W], 范围[0,1]
    返回:
        features: 提取的特征向量 [1,C]
    """
    # VGG预处理: [0,1] -> ImageNet标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(frame_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(frame_tensor.device)
    normalized_frame = (frame_tensor - mean) / std
    
    # 提取特征
    features = vgg_model(normalized_frame)
    features = features.mean(dim=[2, 3])
    
    return features

class VGGBackbone:
    def __init__(self, device):
        self.device = device
        
        # --- 逻辑来自原 set_up_vgg_extractor ---
        # 1. 加载预训练 VGG16
        vgg16 = models.vgg16(pretrained=True)
        self.feature_extractor = vgg16.features
        
        # 2. 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. 移动到设备并设置 eval
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()
        
        # 4. 定义预处理 (Resize 224 + Normalize)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, frame_bgr):
        """
        对应原 extract_vgg_feature 函数
        输入: cv2 读取的 BGR 图片 (numpy array)
        输出: [1, 512, 1, 1] 的 Tensor
        """
        # OpenCV (BGR) -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # 转换为 PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # 应用转换 (Resize -> Tensor -> Norm)
        input_tensor = self.preprocess(pil_image)
        
        # 添加 batch 维度
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            features = self.feature_extractor(input_batch)
            pooled_features = self.global_avg_pool(features) # .squeeze logic 可以放在这里，也可以外面
            
        # 保持维度为 [1, 512, 1, 1] 以便后续处理，或者根据需要 squeeze
        return pooled_features

    # Backward-compat alias (some earlier integrations used `encode()` naming).
    def encode(self, frame_bgr):
        return self.extract(frame_bgr)
