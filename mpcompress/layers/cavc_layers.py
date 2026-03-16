import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalEmbedding(nn.Module):
    """正余弦位置编码 (用于 mod 嵌入)"""
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalEmbedding 维度 {dim} 必须是偶数。")
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: 1D tensor of integers (mod values), shape [B]
        """
        device = x.device
        half_dim = self.dim // 2
        # 计算 log(10000) / (half_dim - 1)
        emb = math.log(10000) / (half_dim - 1)
        # 计算 1 / (10000^(2i / dim))
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 广播 x 和 emb
        # x: [B] -> [B, 1]
        # emb: [half_dim] -> [1, half_dim]
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0) 
        # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # [B, dim]
        return emb
    
class MixFeatureProjector(nn.Module):
    def __init__(self, prompt_dim, feat_dim, out_dim) -> None:
        super().__init__()
        self.prompt_embed = SinusoidalEmbedding(prompt_dim)
        mlp_in_dim = prompt_dim + feat_dim
        # 将组合向量 [B, mlp_in_dim] 投影到 [B, features]
        self.condition_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, out_dim * 4),
            nn.ReLU(),
            nn.Linear(out_dim * 4, out_dim)
        )

    def forward(self, mod, feat):
        B = feat.shape[0]
        device = feat.device
        
        # (Req 2) 1. 获取 mod 的嵌入
        # (*** MODIFIED ***) 将输入的数字 mod 转换为 [B] 形状的 tensor
        mod_tensor = torch.full((B,), mod, dtype=torch.long, device=device)
        
        prompt_vec = self.prompt_embed(mod_tensor) # [B, sin_embed_dim]


        if feat is None:
            raise ValueError("未提供 feat 向量。")
        # 确保 feat 是 [B, feat_dim]
        if feat.dim() > 2: feat = feat.squeeze(-1).squeeze(-1) # 降维
        
        # 确保 batch size 一致 (如果 prompt_vec 是 [1, dim] 而 feat 是 [B, dim])
        if prompt_vec.shape[0] != feat.shape[0] and prompt_vec.shape[0] == 1:
                prompt_vec = prompt_vec.expand(feat.shape[0], -1)

        combined_vec = torch.cat([prompt_vec, feat], dim=1) # [B, sin_embed_dim + feat_dim]
        
        # 3. 投影到 U-Net 内部维度
        condition_emb = self.condition_mlp(combined_vec) # [B, features]

        return condition_emb


class ConditionFusionLayer(nn.Module):
    """条件融合模块（特征拼接）"""
    def __init__(self, in_channels, condition_dim):
        """
        Args:
            in_channels (int): U-Net特征图的通道数
            condition_dim (int): 融合后的条件向量维度
        """
        super().__init__()
        
        # 空间投影层 (将条件向量投影到 in_channels)
        self.zero_conv = nn.Conv2d(condition_dim, in_channels, kernel_size=1)

        # 零初始化
        self._zero_init()
    
    def _zero_init(self):
        """零初始化权重"""
        nn.init.zeros_(self.zero_conv.weight)
        if self.zero_conv.bias is not None:
            nn.init.zeros_(self.zero_conv.bias)

        
    def forward(self, x, condition_emb):
        """
        Args:
            x (torch.Tensor): [B, C, H, W] 输入特征图
            condition_emb (torch.Tensor): [B, condition_dim] 条件向量
        """
        # 空间条件注入（拼接方式）
        B, C, H, W = x.shape
        
        # 将条件向量投影并扩展到空间维度
        condition_feat = self.zero_conv(
            condition_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)
        )
        
        return x + condition_feat
    

class CondUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, 
                 sin_embed_dim=128, cond_dim=256, feat_cond=True):
        """
        Args:
            features (int): U-Net 基础通道数
            sin_embed_dim (int): (未使用) 仅用于兼容性
            feat_dim (int): (未使用) 仅用于兼容性
            feat_cond (bool): (未使用) 仅用于兼容性
        """
        super().__init__()
        # self.feat_cond = feat_cond (不再需要)
        
        # (*** MODIFIED ***)
        # 根据您的原始代码，条件向量的维度 (condition_dim) 
        # 被假定为等于 'features'。
        # (来自您 decoder1 中的注释：ConditionFusionLayer(features, features))
        self.condition_dim = cond_dim

        # 编码器（保持不变）
        self.encoder1 = self.contracting_block(in_channels, features)
        self.encoder2 = self.contracting_block(features, features*2)
        self.encoder3 = self.contracting_block(features*2, features*4)
        self.encoder4 = self.contracting_block(features*4, features*8)
        
        # (*** MODIFIED ***) 中间层（添加融合）
        self.middle = nn.Sequential(
            self.contracting_block(features*8, features*16),
            ConditionFusionLayer(features*16, self.condition_dim)
        )
        
        # (*** MODIFIED ***) 解码器（在所有层添加融合）
        self.decoder4 = nn.Sequential(
            self.expansive_block(features*16, features*8),
            ConditionFusionLayer(features*8, self.condition_dim)
        )
        self.decoder3 = nn.Sequential(
            self.expansive_block(features*8 * 2, features*4),
            ConditionFusionLayer(features*4, self.condition_dim)
        )
        self.decoder2 = nn.Sequential(
            self.expansive_block(features*4 * 2, features*2),
            ConditionFusionLayer(features*2, self.condition_dim)
        )
        self.decoder1 = nn.Sequential(
            self.expansive_block(features*2 * 2, features),
            ConditionFusionLayer(features, self.condition_dim) 
        )
        
        # 输出层 (保持不变)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def contracting_block(self, in_channels, out_channels, kernel_size=4, padding=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block
    
    def expansive_block(self, in_channels, out_channels, kernel_size=4, padding=1):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def pad_tensor(self, tensor, target_height, target_width):
        """
        动态填充或裁剪张量，以匹配目标高度和宽度。
        """
        _, _, h, w = tensor.size()
        pad_h = target_height - h
        pad_w = target_width - w
        if pad_h < 0 or pad_w < 0:
            # 如果解码器输出尺寸大于目标尺寸，则裁剪
            tensor = tensor[:, :, :target_height, :target_width]
        elif pad_h > 0 or pad_w > 0:
            # 如果解码器输出尺寸小于目标尺寸，则填充
            # 使用 ReplicationPad2d 进行填充
            pad_layer = nn.ReplicationPad2d((0, pad_w, 0, pad_h))
            tensor = pad_layer(tensor)  # [C, H + pad_h, W + pad_w]
        return tensor

    def forward(self, x, feat):
        """
        Args:
            x (torch.Tensor): 输入图像 [B, 3, H, W]
            feat (torch.Tensor): 条件向量 [B, condition_dim]
                (由外部的 MixFeatureProjector 生成)
        """
        
        # 编码路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # (*** MODIFIED ***) 中间层
        m = self.middle[0](e4)           # contracting_block
        m = self.middle[1](m, feat)     # ConditionFusionLayer
        
        # (*** MODIFIED ***) 解码路径
        d4 = self.decoder4[0](m)           # expansive_block
        d4 = self.decoder4[1](d4, feat)    # ConditionFusionLayer
        d4 = self.pad_tensor(d4, e4.size(2), e4.size(3))
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.decoder3[0](d4)           # expansive_block
        d3 = self.decoder3[1](d3, feat)    # ConditionFusionLayer
        d3 = self.pad_tensor(d3, e3.size(2), e3.size(3))
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.decoder2[0](d3)           # expansive_block
        d2 = self.decoder2[1](d2, feat)    # ConditionFusionLayer
        d2 = self.pad_tensor(d2, e2.size(2), e2.size(3))
        d2 = torch.cat([d2, e2], dim=1) 
        
        d1 = self.decoder1[0](d2)           # expansive_block
        d1 = self.decoder1[1](d1, feat)     # ConditionFusionLayer (和原来一样)
        d1 = self.pad_tensor(d1, e1.size(2), e1.size(3))
        d1 = torch.cat((d1, e1), dim=1)
        
        return self.final_conv(d1)
    

if __name__ == "__main__":
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = 4
    H, W = 256, 256
    FEAT_DIM = 256
    SIN_EMBED_DIM = 128
    
    print("--- 测试 (feat_cond=True) ---")
    model_with_feat = CondUNetGenerator(
        in_channels=3, 
        out_channels=3, 
        features=64, 
        sin_embed_dim=SIN_EMBED_DIM, 
        feat_dim=FEAT_DIM, 
        feat_cond=True
    ).to(device)
    
    # 打印可训练参数 (检查 'mod_prompt' 是否已移除)
    print("--- 检查模型参数 ---")
    for name, param in model_with_feat.named_parameters():
        if 'mod_prompt' in name:
            print(f"警告：'mod_prompt' 仍然存在！")
        # print(f"{name}: requires_grad={param.requires_grad}")
    print("参数检查完毕。\n")


    # 模拟输入
    x = torch.randn(B, 3, H, W).to(device)
    # (*** MODIFIED ***) mod 现在是一个 int
    mod = 5 # 假设使用模式 5
    feat_vec = torch.randn(B, FEAT_DIM).to(device) # 特征向量
    
    print(f"Input x shape: {x.shape}")
    print(f"Input mod (int): {mod}")
    print(f"Input feat_vec shape: {feat_vec.shape}")
    
    # 前向传播
    try:
        output = model_with_feat(x, mod, feat=feat_vec)
        print(f"Output shape: {output.shape}")
        assert output.shape == (B, 3, H, W)
        print("Model (feat_cond=True) test successful!")
        
    except Exception as e:
        print(f"Model test failed: {e}")

    print("\n--- 测试 (feat_cond=False) ---")
    model_no_feat = CondUNetGenerator(
        in_channels=3, 
        out_channels=3, 
        features=64, 
        sin_embed_dim=SIN_EMBED_DIM, 
        feat_dim=FEAT_DIM, # 即使提供了，也不会使用
        feat_cond=False
    ).to(device)

    # 前向传播 (feat=None)
    try:
        output_no_feat = model_no_feat(x, mod, feat=None)
        print(f"Output shape (no feat): {output_no_feat.shape}")
        assert output_no_feat.shape == (B, 3, H, W)
        print("Model (feat_cond=False) test successful!")
    
    except Exception as e:
        print(f"Model test failed: {e}")