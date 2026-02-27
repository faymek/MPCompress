#!/bin/bash

# use poetry python env
poetry_env=$(poetry env info -p)
source "$poetry_env/bin/activate"
PROJECT_ROOT=/home/faymek/MPCompress

export CUDA_VISIBLE_DEVICES=0

# === 固定层 ===
LAYER=6
LAMBDAS=(1 2 10 20 50)

# === 进入项目主目录 ===
for LAMBDA in "${LAMBDAS[@]}"
do
    LOG_FILE="${PROJECT_ROOT}/logs/layer${LAYER}_lambda${LAMBDA}.txt"
    echo "▶▶▶ Logging to $LOG_FILE"

    {
        # echo "========== Training: Layer ${LAYER}, λ=${LAMBDA} =========="
        # cd $PROJECT_ROOT/examples/vitdet
        # python examples/vitdet/offline/train_hyper.py --layer $LAYER --lambda $LAMBDA

        # echo ""
        # echo "========== Evaluation: Layer ${LAYER}, λ=${LAMBDA} =========="
        # python examples/vitdet/offline/eval_hyper.py --layer $LAYER

        echo ""
        echo "========== Testing: Layer ${LAYER}, λ=${LAMBDA} =========="
        COMPRESS_LAYER=$LAYER \
            python examples/vitdet/test_hyper.py \
            examples/vitdet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
            weights/vitdet/pretrained/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
        cd ..

        echo ""
        echo "✅ Completed Layer ${LAYER} λ=${LAMBDA}"
        echo "--------------------------------------"
    } 2>&1 | tee "$LOG_FILE"
done

