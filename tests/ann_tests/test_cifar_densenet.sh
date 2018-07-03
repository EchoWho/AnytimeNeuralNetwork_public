#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_cifar_densenet/$TIME
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_cifar_densenet/$TIME
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR

python ${CONFIG_DIR}/${ANN_APP_DIR}/densenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
-n=16 -g=16 -s=3 --densenet_version=dense \
--reduction_ratio=0.5 \
--ds_name=cifar100 --batch_size=64 --nr_gpu=1 --samloss=100 --adaloss_gamma=0.14 --adaloss_momentum=0.99 --adaloss_final_extra=1.0 --adaloss_update_per=100 --sum_rand_ratio=0 --is_select_arr -f=5
