#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/ann_policy/train_policy
MODEL_DIR=${GLOBAL_MODEL_DIR}/ann_policy/train_policy
CONFIG_DIR=.

mkdir -p $MODEL_DIR

python $CONFIG_DIR/examples/AnytimeNetwork/ann_policy.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--batch_size=16 \
--is_reg \
