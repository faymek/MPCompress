# 代码使用说明

本代码对 ViT-Det 预训练的 ViT-B/16 模型进行特征编码，包括特征提取、编码解码、重载和下游任务评测。

**特征提取：**在 COCO2017 验证集 5000 张图片上提取 layer3、layer6、layer9、layer12 四个分割点的特征。

**特征编码：**对中间层特征采用 ffmpeg libx264、ffmpeg libx265 和超先验网络三种特征编码方法。

**特征重载和下游任务评测：**对解码后特征进行重载，在 COCO2017 验证集上进行目标检测和语义分割任务。

## 实验环境准备

### 环境配置

请参考 README.md 中的说明，先配置 Poetry 环境。

激活 Poetry 环境：
```sh
poetry shell
```

### 数据集

COCO2017 验证集 val2017.zip 和标注文件 annotations_trainval2017.zip，解压到 data/coco/annotations 和 data/coco/val2017

### 预训练模型

1. 下载预训练的 ViT-B/16 模型权重到 weights/vitdet 文件夹。下载链接：https://disk.pku.edu.cn/link/AA0BC7532D38224BE1A31D0BFB3C68FAF7

2. 下载超先验特征编码器预训练权重到 weights/vitdet 文件夹：https://disk.pku.edu.cn/link/AAD2752F2CB19E4B3F961F01E29450E4A8

最后的文件结构：

```
weights/vitdet
  pretrained/
    xxx-e15fe294.pth
  checkpoints/
    layrer3/
    ...
```

## 测试

测试 ffmpeg 编码

```shell
CUDA_VISIBLE_DEVICES=0 python examples/vitdet/test_ffmpeg.py \
examples/vitdet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
weights/vitdet/pretrained/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
```

测试 hyperprior 编码

```shell
CUDA_VISIBLE_DEVICES=0 python examples/vitdet/test_hyper.py \
examples/vitdet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
weights/vitdet/pretrained/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
```

批量测试 ffmpeg

```shell
bash examples/vitdet/run_codec.sh
```

批量测试 hyperprior

```shell
bash examples/vitdet/run_hyper.sh
```

## 训练超先验特征编码器

首先提取特征

```shell
CUDA_VISIBLE_DEVICES=0 python examples/vitdet/extract_feature.py \
examples/vitdet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
weights/vitdet/pretrained/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
```

然后训练超先验特征编码器

```shell
bash examples/vitdet/train_hyper.sh
```