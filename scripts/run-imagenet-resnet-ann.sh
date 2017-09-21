#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# Note that $s need to be adjusted as well. (d, s):
# (14, 1), (26, 2), (50, 2), (101, 3), (18, 2), (34, 2), (152, 5)
d=26  
s=2
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/imagenet-ann$d/
MODEL_DIR=${GLOBAL_MODEL_DIR}/imagenet_model/ann$d/
CONFIG_DIR=.

mkdir -p $MODEL_DIR

# Run the actual job
python $CONFIG_DIR/examples/AnytimeNetwork/imagenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
-f=5 \
--opt_at=-1 \
-d=$d \
--nr_gpu=2 \
--batch_size=64 \
--samloss=6 \
-c=64 \
-s=$s \
--num_classes=1000 \
--init_lr=0.0125 \
--load=${MODEL_DIR}/checkpoint
