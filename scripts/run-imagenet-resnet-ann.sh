#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# Note that $s need to be adjusted as well. (d, s):
# (14, 1), (26, 2), (50, 2), (101, 3), (18, 2), (34, 2), (152, 5)
d=26  
s=2
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/imagenet-ann${d}-f${f}
MODEL_DIR=${GLOBAL_MODEL_DIR}/imagenet_model/ann${d}-f${f}
CONFIG_DIR=.

mkdir -p $MODEL_DIR

# Run the actual job
python $CONFIG_DIR/anytime_models/examples/imagenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
-d=50 -s=2 --batch_size=64 --nr_gpu=1 -f=5 --samloss=100 --exp_gamma=0.07 --sum_rand_ratio=0 --is_select_arr --opt_at=-1 -c=64 --num_classes=1000 \
--evaluate=val
