#!/bin/bash

# use poetry python env
poetry_env=$(poetry env info -p)
source "$poetry_env/bin/activate"
PROJECT_ROOT=/home/faymek/MPCompress

export CUDA_VISIBLE_DEVICES=0

# === 固定层 ===
LAYER=12
CODECS=("libx264")
QPS=(12 22 32 42)
PRESETS=("placebo")

# === 进入项目主目录 ===
for CODEC in "${CODECS[@]}"; do
  for QP in "${QPS[@]}"; do
    for PRESET in "${PRESETS[@]}"; do
      LOG_FILE="${PROJECT_ROOT}/logs/layer${LAYER}_${CODEC}_Qp${QP}_${PRESET}.txt"
      echo "▶▶▶ Logging to $LOG_FILE"
      {
        echo "========== Evaluation: Layer ${LAYER}, ${CODEC}, Qp${QP}, ${PRESET} =========="
        COMPRESS_LAYER=$LAYER CODEC=$CODEC QP=$QP PRESET=$PRESET \
          python examples/vitdet/test_ffmpeg.py \
          examples/vitdet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
          weights/vitdet/pretrained/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
        cd ..
        echo ""
        echo "✅ Completed Layer ${LAYER}, ${CODEC}, Qp${QP}, ${PRESET}"
        echo "--------------------------------------"
      } 2>&1 | tee "$LOG_FILE"
    done
  done
done

