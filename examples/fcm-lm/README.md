# FCM-LM

当前专注复现论文 **“Feature Coding in the Era of Large Models”** 中的 **fcm‑lm** 基线实验（基于 DINOv2 的特征压缩）。

---

## 目录结构总览

```
MPCompress/
├─ .venv/                                # 虚拟环境
│
├─ data/
│  └─ dataset/                           # 论文实验所用各数据子集
│      ├─ ImageNet_val_sel100/
│      ├─ VOC2012/                       # 完整 VOC2012
│      └─ VOC2012_sel20/                 # 论文选取的 20 张子集
│
├─ models/                               # 预训练权重与下载脚本
│  ├─ backbone/
│  │   └─ dinov2_vitg14_pretrain.pth
│  ├─ clf_head/
│  │   └─ dinov2_vitg14_linear_head.pth
│  ├─ seg_head/
│  │   └─ dinov2_vitg14_voc2012_linear_head.pth
│  └─ download_pretrained.sh
│
├─ self_extract/                         # 自行提取的特征
│
├─ test-fc/                              # 评估阶段解码后的特征
│  ├─ ImageNet_val_sel100--dinov2_cls/
│  ├─ ImageNet--dinov2_cls/
│  ├─ VOC2012_sel20--dinov2_seg/
│  └─ VOC2012_val_sel20--dinov2_seg/
│
├─ train-fc/
│  └─ ImageNet--dinov2_cls/
│      ├─ feat/
│      └─ hyperprior/
│
├─ examples/
│  └─ fcm-lm/                            # 三条基线脚本及配置
│      ├─ conf/                          # YAML 任务配置
│      │   ├─ hyper_dinov2_cls.yaml
│      │   ├─ hyper_dinov2_dpt.yaml
│      │   ├─ hyper_dinov2_seg.yaml
│      │   ├─ hyper_llama3_csr.yaml
│      │   ├─ hyper_sd3_tti.yaml
│      │   ├─ vtm_dinov2_cls.yaml
│      │   ├─ vtm_dinov2_dpt.yaml
│      │   ├─ vtm_dinov2_seg.yaml
│      │   ├─ vtm_llama3_csr.yaml
│      │   └─ vtm_sd3_tti.yaml
│      ├─ conf-mmcv/                     # MMCV-style 模型结构文件
│      │   └─ dinov2_vitg14_voc2012_linear_config.py
│      │
│      ├─ run_dinov2-cls_vtm_cls.py
│      ├─ run_dinov2-cls_hyper_cls.py
│      └─ run_dinov2-seg_vtm_seg.py
│
└─ README.md
```

**约定**  
- 项目根目录为 `MPCompress/`。  
- 文档中的路径均以此根目录为参考点。  

---

## 环境配置

```zsh
# 仅首次安装 Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 克隆并进入仓库
git clone https://github.com/xxx/MPCompress.git
cd MPCompress

# 一键安装依赖并创建隔离虚拟环境
poetry install
poetry run pip install --editable .
```

CUDA / GCC 要满足 `mmcv==2.1.0` 的编译要求，请提前检查
`nvcc --version` 与 `gcc --version`。

---

## 依赖组件（MMCV & MMSegmentation）

### 编译安装 `mmcv==2.1.0`

```zsh
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout v2.1.0
pip install -r requirements/optional.txt
python setup.py build_ext
python setup.py develop
python .dev_scripts/check_installation.py
```

### 安装 `mmsegmentation==1.2.1`

```zsh
pip install -U openmim
mim install "mmsegmentation==1.2.1"
```

---

## 配置文件与自动加载

| 作用 | 路径 | 说明 |
|------|------|------|
| **Feature-Coding** 超参数 | `conf/hyper_dinov2_cls.yaml`、`conf/vtm_dinov2_cls.yaml`、`conf/vtm_dinov2_seg.yaml` | 控制量化 bit-depth、截断比例等；可复制并通过 `--cfg` 指定 |
| **DINOv2** 分割 **MMCV** 测试配置 | `conf-mmcv/dinov2_vitg14_voc2012_linear_config.py` | 定义 DINOv2 backbone 与线性分割头的测试流程，包含数据加载、滑窗策略与预处理设置，适用于 VOC2012 数据集测试；更换路径或类别需同步调整 |

**自动加载机制**  
- 若脚本未显式指定 `--cfg`（Feature-Coding 超参数），会根据任务名自动匹配同名 YAML 模板。  
- 如需自定义，复制模板并在 CLI 中传入 `--cfg path/to/your.yaml` 即可覆盖。  
- 分割任务使用的 MMCV 配置默认读取 `conf-mmcv/dinov2_vitg14_voc2012_linear_config.py`；如需自定义，可在脚本中修改 `--mmcv_cfg` 指向自己的 `.py` 配置文件。  


---

## 数据与模型准备

| 任务 | 必要原始数据集 | 论文选取子集/列表 | 在项目中的放置路径 |
|------|--------------|------------------|-------------------|
| 图像分类 | ImageNet | `ImageNet_val_sel100/`（100 张）| `data/dataset/ImageNet_val_sel100/` |
| 语义分割 | VOC2012 | `VOC2012_sel20/`（20 张） | `data/dataset/VOC2012_sel20/` |

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

| 任务           | 指标 |
|------|------|
| **图像分类**   | BPFP、MSE、Accuracy |
| **语义分割**   | BPFP、MSE、mIoU |

---
