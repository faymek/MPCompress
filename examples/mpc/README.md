# MPC 评估代码

这个评估系统提供了完整的MPC模型评估功能，一次编码，返回多个任务所需的特征，分别进行评估。

- 图像重建任务：PSNR, MS-SSIM, LPIPS, CLIP-SIM, FID
- 图像分类任务：Top-1 Accuracy, Top-5 Accuracy
- 语义分割任务：mIoU
- 压缩效率：BPP, 编码时间, 解码时间


## 配置环境

按照项目 README.md 中的说明配置环境。

你可能需要设置 HuggingFace 的镜像地址，以更快地加载模型。
```
export HF_ENDPOINT=https://hf-mirror.com 
```

## 测试方法

```bash
# ImageNet分类任务
python examples/mpc/run_mpc_eval.py \
    --config examples/mpc/eval_config.yaml \
    --checkpoint /path/to/model.pth \
    --task imagenet_sel100_cls \
    --quality 12.0 \
    --cuda --verbose --real

# VOC2012分割任务 
python examples/mpc/run_mpc_eval.py \
    --config examples/mpc/eval_config.yaml \
    --checkpoint /path/to/model.pth \
    --task voc2012_sel20_seg \
    --quality 12.0 \
    --cuda --verbose --real
```

参数说明：

- `--config`: 配置文件路径
- `--checkpoint`: 模型权重路径
- `--task`: 任务名称，需要与配置文件中的任务名称一致
- `--quality`: 质量因子，仅用作任务标签
- `--cuda`: 使用CUDA
- `--verbose`: 启用详细输出，打印每个文件的评估结果
- `--real`: 启用真实熵编码，写入码流；否则使用码率估计，不写入码流


## MPC 实现说明

当前的 MPC 模型实现了图像编码的两层编码

- 第一层采用VQGAN，具体为 [VQGAN-Compression](https://github.com/CUC-MIPG/VQGAN-Compression) 提供的预训练模型，该模型对原始VQGAN进行了聚类微调，将VQ码本大小降低到1024，从而允许更低的码率。
- 第二层采用DINOv2，具体为 [timm](https://github.com/huggingface/timm) 库的实现，该实现允许任意分辨率输入和任意patch_size的处理。

MPC 模型实现了如下主要的类方法

- forward: 前向推理，用于训练，返回训练所需的特征
- forward_test: 前向推理，用于测试，按需返回下游任务所需的特征
- compress: 实际编码方法，返回码流的中间表示
- decompress: 实际解码方法，按需返回下游任务所需的特征


## 数据集

所有数据集都返回 `(img, img_meta)` 格式：

```python
img_meta = {
    "img_path": "图像文件路径",
    "img_name": "图像文件名（不含扩展名）",
    "ori_size": "原始图像尺寸",
    "target": "分类标签（可选）",
    "seg_label_path": "分割标签路径（可选）"
}
```

## 评估指标

### 图像质量指标

图像评估指标，建议采用 `pyiqa==0.1.13` 中提供的多种 metrics，以统一计算。

- **PSNR**: 峰值信噪比，值越高表示质量越好
- **MS-SSIM**: 多尺度结构相似性，范围[0,1]，值越高表示质量越好
- **LPIPS**: 学习型感知图像质量评估，值越低表示质量越好
- **CLIP-SIM**: CLIP模型计算的相似度，值越高表示语义相似度越高
- **FID**: 计算生成图像与真实图像之间的分布距离，值越低表示质量越好

### 分类指标
- **Top-1 Accuracy**: 预测的最高置信度类别与真实标签匹配的比例
- **Top-5 Accuracy**: 真实标签在预测的前5个类别中的比例

### 分割指标
- **mIoU**: 平均交并比，计算所有类别的IoU平均值

### 压缩效率指标
- **BPP**: 每像素比特数，表示压缩率
- **编码时间**: 压缩一张图像所需的时间
- **解码时间**: 解压缩一张图像所需的时间


