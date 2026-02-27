# RFC/MLoRE 多任务特征压缩示例

本示例展示如何使用 MPCompress 框架进行 MLoRE (Multi-task Low-Rank Expert) 多任务特征压缩的训练和评估。

## 框架规范

本实现严格遵循 [MPCompress Framework](https://faymek.github.io/MPCompress/framework/) 设计规范：

- **DataUnitCodec**: `compress(x)` → `coded_unit`, `decompress(coded_unit)` → `task_feats`
- **FrameCodec**: 处理单帧多任务压缩
- **VideoCodec**: 处理视频序列压缩

## 目录结构

```
examples/rfc/
├── config/
│   ├── eval_base.yaml          # 评估基础配置
│   ├── eval_rfc_pascal.yaml    # PASCAL Context 评估配置
│   ├── eval_rfc_nyud.yaml      # NYUD 评估配置
│   ├── train_stage0.yaml       # Stage 0: 无压缩预训练
│   ├── train_stage1.yaml       # Stage 1: 压缩模块训练
│   └── train_stage2.yaml       # Stage 2: Mona 微调
├── run_eval_rfc.py             # 评估脚本
├── run_train_rfc.py            # 训练脚本
└── README.md                   # 本文档
```

## 支持的数据集

### PASCAL Context
支持 5 种任务：
- 语义分割 (semseg): 21 类
- 边缘检测 (edge): 二值
- 表面法向量 (normals): 3 通道
- 显著性检测 (sal): 二值
- 人体部位分割 (human_parts): 7 类

### NYUD
支持 5 种任务：
- 语义分割 (semseg): 40 类
- 边缘检测 (edge): 二值
- 表面法向量 (normals): 3 通道
- 深度估计 (depth): 单通道
- 场景分类 (scene): 13 类

## 环境配置

按照项目根目录 `README.md` 配置环境后，确保以下依赖可用：

```bash
# 安装依赖
pip install easydict omegaconf tensorboard tqdm
```

## 数据准备

### PASCAL Context

下载数据集并解压到指定目录：
```
/path/to/PASCALContext/
├── JPEGImages/
├── pascal-context/
├── semseg/
├── human_parts/
├── normals_distill/
├── sal_distill/
└── ImageSets/
```

### NYUD

下载数据集并解压到指定目录：
```
/path/to/NYUD_MT/
├── images/
├── segmentation/
├── edge/
├── normals/
├── depth/
└── gt_sets/
```

## 训练流程

MLoRE 采用三阶段训练：

### Stage 0: 无压缩预训练

训练通用的多任务 backbone，不包含压缩模块。

```bash
# 单卡训练
python examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage0.yaml \
    --run_mode train

# 多卡分布式训练
torchrun --nproc_per_node=4 examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage0.yaml \
    --run_mode train
```

### Stage 1: 压缩模块训练

加载 Stage 0 模型，仅训练压缩网络。需要修改配置文件中的 `load_multitask_encoder` 路径。

```bash
# 修改 train_stage1.yaml 中的 load_multitask_encoder 路径
# load_multitask_encoder: /path/to/stage0/checkpoint.pth.tar

torchrun --nproc_per_node=4 examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage1.yaml \
    --run_mode train
```

**注意**: Stage 1 每次只训练一个任务的压缩头。需要修改配置文件中的 `task_dictionary` 来选择任务。

### Stage 2: Mona Adapter 微调

加载 Stage 1 模型，微调 Mona adapter 和解码头。

```bash
# 修改 train_stage2.yaml 中的 checkpoint 路径
# checkpoint: /path/to/stage1/checkpoint.pth.tar

torchrun --nproc_per_node=4 examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage2.yaml \
    --run_mode train
```

## 评估

### 评估单个任务

```bash
# PASCAL Context 语义分割
python examples/rfc/run_eval_rfc.py \
    --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_pascal.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --task pascal_semseg \
    --cuda --real \
    --output_dir ./eval_results/pascal_semseg

# PASCAL Context 边缘检测
python examples/rfc/run_eval_rfc.py \
    --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_pascal.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --task pascal_edge \
    --cuda --real \
    --output_dir ./eval_results/pascal_edge
```

### 评估多任务

```bash
# PASCAL Context 全部任务
python examples/rfc/run_eval_rfc.py \
    --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_pascal.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --task pascal_multitask \
    --cuda --real \
    --output_dir ./eval_results/pascal_multitask

# NYUD 全部任务
python examples/rfc/run_eval_rfc.py \
    --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_nyud.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --task nyud_multitask \
    --cuda --real \
    --output_dir ./eval_results/nyud_multitask
```

### 评估参数说明

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径，可指定多个叠加 |
| `--checkpoint` | 模型检查点路径 |
| `--task` | 评估任务名称（需与配置文件中定义一致） |
| `--cuda` | 使用 CUDA |
| `--real` | 使用实际熵编码（写入码流）；否则使用码率估计 |
| `--verbose` | 详细输出每张图像的评估结果 |
| `--output_dir` | 结果保存目录 |

## 评估指标

### 压缩效率指标
- **BPP**: 每像素比特数
- **编码时间**: 压缩一张图像所需时间
- **解码时间**: 解压缩一张图像所需时间

### 任务性能指标

| 任务 | 指标 | 说明 |
|------|------|------|
| semseg | mIoU | 平均交并比 |
| edge | ODS, OIS | 最优数据集/图像尺度 F-measure |
| normals | Mean, Median | 角度误差 |
| sal | maxF, MAE | 最大 F-measure, 平均绝对误差 |
| human_parts | mIoU | 平均交并比 |
| depth | RMSE, Abs Rel | 均方根误差, 相对误差 |
| scene | Accuracy | 分类准确率 |

## 模型接口

### MLoREFrameCodec

```python
from mpcompress.models import MLoREFrameCodec

# 创建模型
model = MLoREFrameCodec(
    stage='stage1',  # 'stage0', 'stage1', 'stage2'
    img_size=(512, 512),
    pretrained=True,
)

# 前向推理（训练）
out = model(x, tasks=['semseg', 'edge'])
# out: {'semseg': tensor, 'edge': tensor, 'bpp_loss': tensor, 'mse_loss': tensor}

# 压缩
coded_unit = model.compress(x, tasks=['semseg', 'edge'])
# coded_unit: {'strings': {...}, 'pstate': {...}}

# 解压缩
task_feats = model.decompress(coded_unit)
# task_feats: {'semseg': tensor, 'edge': tensor}
```

### MLoREVideoCodec

```python
from mpcompress.models import MLoREVideoCodec

# 创建视频编解码器
codec = MLoREVideoCodec(stage='stage1', img_size=(512, 512))

# 压缩视频
coded_data = codec.compress_video(video_reader, meta)
# coded_data: {'type': 'frame_wise_video', 'data': {...}, 'meta': {...}}

# 解压缩视频
results = codec.decompress_video(coded_data)
# results: {0: {'semseg': tensor, ...}, 1: {...}, ...}
```

## 配置文件说明

配置文件采用 YAML 格式，支持多文件叠加。

### 主要配置项

```yaml
# 模型配置
model:
  type: MLoREFrameCodec
  stage: stage1
  img_size: [512, 512]
  final_embed_dim: 640

# 任务配置
tasks:
  NAMES: [semseg, edge, normals, sal, human_parts]
  NUM_OUTPUT:
    semseg: 21
    edge: 1
    normals: 3
    sal: 2
    human_parts: 7

# 损失权重
loss_kwargs:
  loss_weights:
    semseg: 1.0
    edge: 50.0
    normals: 10.0
    bpp_loss: 1.0
    mse_loss: 100.0
```

## 引用

如果您使用了本代码，请引用：

```bibtex
@article{mlore,
  title={MLoRE: Multi-task Low-Rank Expert for Feature Compression},
  author={...},
  journal={...},
  year={2024}
}
```

## 常见问题

### Q: 如何修改数据集路径？
A: 修改配置文件中 `datasets` 部分的 `root` 路径，或在 `db_paths` 中配置。

### Q: 如何切换训练任务？
A: 修改配置文件中 `task_dictionary` 的 `include_xxx` 字段，以及 `TASKS.NAMES` 列表。

### Q: 如何使用预训练权重？
A: 
- Stage 0: 自动使用 ImageNet 预训练
- Stage 1/2: 设置 `load_multitask_encoder` 或 `checkpoint` 路径

### Q: 分布式训练失败？
A: 检查 NCCL 环境变量设置，确保所有节点网络连通。




