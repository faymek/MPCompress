# MPC 评估代码

[DCVC-RT](https://github.com/microsoft/DCVC) 是第一个实现100+ FPS 1080p编码和4K实时编码的神经视频编解码器（NVC），压缩比与ECM相当。除此之外，DCVC-RT追求更实用的神经视频编解码器解决方案，并支持可变码率、统一RGB和YUV编码等实用功能。

这里选择 DCVC-RT 作为典型的基于NN的视频编码器，并在此基础上撰写视频测试 pipeline。

## 配置环境

首先按照项目 README.md 中的说明配置环境。

其次配置 DCVC-RT 的环境。

```shell
sudo apt-get install cmake g++ ninja-build
poetry activate
cd mpcompress/cpp/
pip install .
cd mpcompress/layers/extensions/inference/
pip install .
```

下载预训练模型：

下载地址：https://1drv.ms/u/s!AozfVVwtWWYoiS5mcGX320bFXI0k?e=iMeykH

这里使用了绝对目录，您可将其替换为您的实际目录：
- /home/faymek/DCVC/checkpoints/cvpr2025_image.pth.tar
- /home/faymek/DCVC/checkpoints/cvpr2025_image.pth.tar

## 测试方法

```bash
# 测试 UVG 重建
python examples/dcvc-rt/run_eval_dcvcrt.py \
    --config examples/dcvc-rt/config/eval_base.yaml examples/dcvc-rt/config/eval_dcvcrt.yaml \
    --checkpoint "" \
    --task uvg_val_rec \
    --head "" \
    --quality 1.0 \
    --cuda --recon 2 --real \
    --output_dir eval_uvg_val_dcvcrt

```

参数说明：

- `--config`: 配置文件路径，可多个叠加
- `--checkpoint`: 模型权重路径
- `--task`: 任务名称，需要与配置文件中的任务名称一致
- `--quality`: 质量因子，仅用作任务标签
- `--cuda`: 使用CUDA
- `--verbose`: 启用详细输出，打印每个文件的评估结果
- `--recon`: 对于MPC模型，使用第几层分支的重建图像，当前可选[0,1,2]
- `--real`: 启用真实熵编码，写入码流；否则使用码率估计，不写入码流


## VideoCodec 实现约定

端到端 Video Codec 应实现的类方法。

- compress_video()
  - init_compress()
  - for x in video_reader:
    - encoded, pstate = compress_frame(x)
    - buff = write_frame_by_syntax(encoded, pstate)
  - close_write()

- decompress_video()
  - init_decompress()
  - for _ in frame_num:
    - bitstream, pstate = read_frame_by_sntax(buff)
    - decoded = decompress_frame(bitstream, pstate)
  - close_read()
  return [
    {"x_hat": xx, "cls": xx, "seg": xx},
    {"x_hat": xx, "cls": xx, "seg": xx}
  ]

约定
- Codec 应至少实现 compress_video 和 decompress_video 方法。解码返回逐帧的重建结果，形成一个列表。
- Codec 可选地实现 compress_frame 和 decompress_frame 方法。解码返回当前帧的重建结果。


## Video 数据集

所有数据集都返回 `(video_reader, video_meta)` 格式：

video_reader 可以支持如下用法：
```
frame = video_reader.read_one_frame()
all_formats = frame.y, frame.u, frame.v, frame.yuv444, frame.rgb
```

video_meta 应包含下面的源信息

```python
video_meta = {
    "seq_name": "Beauty_1920x1080.yuv", # 序列名称
    "src_width": 1920, # 图像宽度
    "src_height": 1080, # 图像高度
    "frame_num": 64， # 帧数
    "cls_labels": [] # 逐帧的分类标签
    "seg_label_paths": [] # 逐帧的分割标签
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


