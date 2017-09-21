#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/imagenet-ann101/
MODEL_DIR=${GLOBAL_MODEL_DIR}/imagenet_model/ann101/
CONFIG_DIR=/home/dedey/AnytimeNeuralNetwork

mkdir -p $MODEL_DIR

# Run the actual job
python $CONFIG_DIR/examples/AnytimeNetwork/imagenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
-f=5 \
--opt_at=-1 \
-d=101 \
--nr_gpu=2 \
--batch_size=64 \
--samloss=6 \
-c=64 \
-s=3 \
--num_classes=1000 \
--init_lr=0.0125 \
--load=${MODEL_DIR}/checkpoint
