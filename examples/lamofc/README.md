# LaMoFC

当前集成了论文 **“Feature Coding in the Era of Large Models” (LaMoFC)** 中的基线实验（基于 DINOv2 的特征压缩）。

## 环境配置

请参考 README.md 中的说明，先配置 Poetry 环境。

激活 Poetry 环境：
```sh
poetry shell
```

## 准备数据

请参考 README.md 中的说明，下载好数据并解压到对应位置。

## 准备 VTM

请下载 VTM 并解压到 `weights/std_codec/` 目录下，并且 checkout 到 VTM-21.0 版本。
```shell
cd weights/std_codec
git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
mv VVCSoftware_VTM VTM-21.0
cd VTM-21.0
git checkout VTM-21.0
```

接下来进行编译
```shell
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
```

## 运行示例

在项目根目录运行：

测试滑动窗口推理

```shell
CUDA_VISIBLE_DEVICES=0 python examples/lamofc/run_eval_slide.py \
    --config examples/lamofc/config/eval_base.yaml examples/lamofc/config/dino_orig_slide_patch_small_last1_vtm.yaml \
    --preset voc2012_sel20_seg \
    --head "voc2012_seg_small_last1" \
    --quality 1.0 \
    --cuda --output_dir eval_test --real
```

测试整图推理

```shell
CUDA_VISIBLE_DEVICES=0 python examples/lamofc/run_eval.py \
    --config examples/lamofc/config/eval_base.yaml examples/lamofc/config/dino_timm_patch_small_last1_vtm.yaml \
    --preset voc2012_sel20_seg \
    --head "voc2012_seg_small_last1" \
    --quality 1.0 \
    --cuda --output_dir eval_test --real

```

更多的 preset 和 head 配置请参考 `examples/lamofc/config/eval_base.yaml`。
