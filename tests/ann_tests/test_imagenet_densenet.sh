#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_imagenet_densenet/$TIME
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_imagenet_densenet/$TIME
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR

python $CONFIG_DIR/${ANN_APP_DIR}/imagenet-dense-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--exp_gamma=0.07 --sum_rand_ratio=0 --is_select_arr -f=5 --samloss=100  --densenet_depth=201 -s=17 --batch_size=24 --nr_gpu=1 --densenet_version=dense --min_predict_unit=10 --reduction_ratio=0.5 --dropout_kp=0.9 --opt_at=-1 -g=32 --num_classes=1000 
