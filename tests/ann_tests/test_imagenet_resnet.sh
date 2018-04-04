#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_imagenet_resnet/$TIME
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_imagenet_resnet/$TIME
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR

python $CONFIG_DIR/${ANN_APP_DIR}/imagenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
-d=50 -c=64 --resnet_version=resnet \
--exp_gamma=0.07 --sum_rand_ratio=0 --is_select_arr -f=5 --samloss=100 \
--batch_size=24 --nr_gpu=1 \
--num_classes=1000 
