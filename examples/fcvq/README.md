# FCVQ

---

## 环境配置

```
# 本项目基于MPCompress实现，参考MPCompress环境配置，额外依赖于CompressAI，建议安装开发者版本，至根目录/code下
git clone https://github.com/InterDigitalInc/CompressAI compressai
cd compressai
pip install -U pip && pip install -e .
```

---

## 配置文件

| 作用                              | 路径                                                         | 说明                                                         |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **FCVQ** 超参数                   | `/code/examples/fcvq/cfg/cls.yaml`、`/code/examples/fcvq/cfg/seg.yaml` | 控制码本维度、码本大小、分块大小、$\lambda$等超参数；可复制并通过 `--cfg` 指定 |
| **DINOv2** 分割 **MMCV** 测试配置 | `/code/examples/fcvq/cfg/dinov2_vitg14_voc2012_linear_config.py` | 定义 DINOv2 backbone 与线性分割头的测试流程，包含数据加载、滑窗策略与预处理设置，适用于 VOC2012 数据集测试；更换路径或类别需同步调整 |
| DINOv2分类测试集配置文件          | `/code/examples/cfg/examples/fcvq/cfg/imagenet_selected_label500.txt` | 从ImageNet数据集中选取的原始图像label                        |
| DINOv2分割测试集配置文件          | `/code/examples/cfg/examples/fcvq/cfg/val_100.txt`           | 从Val2012数据集中选取的原始图像label                         |

- 分割任务使用的 MMCV 配置默认读取 `conf-mmcv/dinov2_vitg14_voc2012_linear_config.py`；如需自定义，可在脚本中修改 `--mmcv_cfg` 指向自己的 `.py` 配置文件。  在测试时直接使用测试集配置文件读取相应的特征数据集。


---

## 数据与模型准备

| 任务     | 必要原始数据集 | 论文选取子集/列表                | 在项目中的放置路径                  |
| -------- | -------------- | -------------------------------- | ----------------------------------- |
| 图像分类 | ImageNet       | `ImageNet_val_sel100/`（100 张） | `data/dataset/ImageNet_val_sel100/` |
| 语义分割 | VOC2012        | `VOC2012_sel20/`（20 张）        | `data/dataset/VOC2012_sel20/`       |

**预训练权重**（已在 `data/models/` 下给出）  
```
data/models/backbone/dinov2_vitg14_pretrain.pth
data/models/clf_head/dinov2_vitg14_linear_head.pth
data/models/seg_head/dinov2_vitg14_voc2012_linear_head.pth
```

若需完整示例，请参考仓库内 `Data_example/` 或上方目录树。

---

## 特征提取流程（训练 / 评估前）

1. **准备数据**  
   将图片及标注放入 `data/dataset/<DATASET_NAME>/`。  
   例如 ImageNet 100 张子集应位于  
   `data/dataset/ImageNet_val_sel100/ILSVRC2012_val_00000001.JPEG` 等。

2. **（可选）按类别随机抽样生成自定义子集**  
   如果希望从完整 ImageNet 中按 *N* 张 / 类快速生成一个轻量验证集，可使用脚本  
   `examples/fcm-lm/sample_imagenet_per_class.py`：  

   ```
   # 仅需修改脚本内 CONFIG 字典，无命令行参数
   python examples/fcm-lm/sample_imagenet_per_class.py
   ```

   - 脚本会：  
     1. 在典型 ImageNet 目录（1000 类子文件夹）中，每类随机抽取 `SAMPLES_PER_CLASS` 张 JPEG；  
     2. 通过 **DINOv2 ViT-G/14** backbone 提取 `[CLS]` 特征并保存为 `.npy`；  
     3. 使用线性分类头得到预测类别并写入 `predictions.txt`。  
   - 关键参数（内部 `CONFIG`）：  

     ```python
     CONFIG = {
         'DATA_ROOT': '/path/to/ImageNet/val',      # 源数据根目录
         'BACKBONE_PATH': '/path/to/dinov2_vitg14_pretrain.pth',
         'HEAD_PATH': '/path/to/dinov2_vitg14_linear_head.pth',
         'OUTPUT_DIR': './outputs',                 # 结果根目录
         'FEATURE_DIR': None,                       # None → OUTPUT_DIR/features
         'SAMPLES_PER_CLASS': 5,                    # <=0 表示该类全部
         'SEED': 42,
     }
     ```

   运行结束后将看到：
   ```
   outputs/
   ├─ features/         # *.npy 特征
   └─ predictions.txt   # “pred_idx & filename”
   ```

---

## 运行示例

```zsh
# 激活虚拟环境
source .venv/bin/activate

# 图像分类 — VTM baseline
CUDA_VISIBLE_DEVICES=0 python examples/fcm-lm/run_dinov2-cls_vtm_cls.py   --cfg examples/fcm-lm/conf/vtm_dinov2_cls.yaml

# 图像分类 — Hyperprior baseline
CUDA_VISIBLE_DEVICES=0 python examples/fcm-lm/run_dinov2-cls_hyper_cls.py   --cfg examples/fcm-lm/conf/hyper_dinov2_cls.yaml

# 语义分割 — VTM baseline
CUDA_VISIBLE_DEVICES=0 python examples/fcm-lm/run_dinov2-seg_vtm_seg.py   --cfg examples/fcm-lm/conf/vtm_dinov2_seg.yaml
```

各脚本会在指定data目录中生成日志并打印指标，在模型训练过程中，会将每个epoch的训练指标打印到相应的日志中。  
仅做推理时，可在 YAML 里关闭训练阶段，或在脚本中设置 `train=False`。

---

## 评估指标

| 任务         | 指标                |
| ------------ | ------------------- |
| **图像分类** | BPFP、MSE、Accuracy |
| **语义分割** | BPFP、MSE、mIoU     |

---

## 集成测试

请按照项目根目录 `README.md` 的说明完成环境配置：安装 Poetry、创建 `.env` 文件（含 `PROJECT_ROOT` 变量），并准备测试数据与权重。

其中 segmentation 任务使用了滑动窗口推理，classification 任务使用了整图推理。

**FCVQ 推理示例**：

```bash 
CUDA_VISIBLE_DEVICES=0 python examples/fcvq/run_eval_slide.py \
    --config examples/fcvq/config/eval_base.yaml examples/fcvq/config/dino_orig_slide_giant_seg_fcvq_64.yaml \
    --preset voc2012_sel20_seg \
    --head voc2012_seg_giant_last1 \
    --quality 1.0 \
    --cuda --output_dir eval_test --real

CUDA_VISIBLE_DEVICES=0 python examples/fcvq/run_eval_slide.py \
      --config examples/fcvq/config/eval_base.yaml examples/fcvq/config/dino_orig_slide_giant_cls_fcvq_512.yaml \
      --preset imagenet_sel100_cls \
      --head imagenet_cls_giant_last1 \
      --quality 1.0 \
      --cuda --output_dir eval_test --real
```

