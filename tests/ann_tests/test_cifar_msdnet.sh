#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_cifar_msdnet/$TIME
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_cifar_msdnet/$TIME
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR

python ${CONFIG_DIR}/${ANN_APP_DIR}/msdensenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--msdensenet_depth=24 \
--ds_name=cifar100 \
--batch_size=128 --nr_gpu=2 \
--samloss=100 -f=5 \
--adaloss_final_extra=1.0 \
--adaloss_update_per=1 \
--sum_rand_ratio=0 \
--is_select_arr \
--adaloss_gamma=0.07 \
--adaloss_momentum=0.9 \
