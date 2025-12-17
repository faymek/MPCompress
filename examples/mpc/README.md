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
# MPC2 DINO Lagre VBR 测试 ImageNet 分类任务
CUDA_VISIBLE_DEVICES=0 python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC2-v3-large-vbr.yaml \
    --preset imagenet_sel2k_cls \
    --head "imagenet_cls_large_last4" \
    --quality 1.0 \
    --cuda --recon 0 --output_dir eval_test --real

# MPC2 DINO Base VBR 测试 VOC2012 分割任务
CUDA_VISIBLE_DEVICES=0 python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC2-v3-base-vbr.yaml \
    --preset voc2012_val_seg \
    --head "voc2012_seg_base_last4" \
    --quality 1.0 \
    --cuda --recon 0 --output_dir eval_test --real

# MPC2 DINO Small VBR 测试 ADE20K 分割任务
CUDA_VISIBLE_DEVICES=0 python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC2-v3-small-vbr.yaml \
    --preset ade20k_val_seg \
    --head "ade20k_seg_small_last4" \
    --quality 1.0 \
    --cuda --recon 0 --output_dir eval_test --real
```

参数说明：

- `--config`: 配置文件路径，可多个叠加
- `--preset`: 预定义的评估任务名称，需要与配置文件中的任务名称一致
- `--head`: 头部模型名称，需要是预定义的头部模型
- `--quality`: 质量因子，仅用作任务标签
- `--cuda`: 使用CUDA
- `--recon`: 对于MPC模型，使用第几层分支的重建图像，当前可选[0,1,2]
- `--real`: 启用真实熵编码，写入码流；否则使用码率估计，不写入码流
