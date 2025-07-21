import torch
import compressai


# 加载 checkpoint
ckpt_path = "/home/liuzk/projects/MPCompress/data/train-fc/ImageNet--dinov2_cls/hyperprior/pretrained_models/trunl-5_trunh5_uniform0_bitdepth1/lambda0.001_epoch800_lr1e-4_bs128_patch256-256_checkpoint.pth.tar"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 获取 state_dict
state_dict = checkpoint["state_dict"]

# 打印包含 entropy_bottleneck 的 key
entropy_keys = [k for k in state_dict.keys() if "entropy_bottleneck" in k]
print(entropy_keys[:10])  # 只看前10个
print(compressai.__version__)
