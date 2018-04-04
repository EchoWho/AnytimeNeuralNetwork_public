#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_cifar_resnet/$TIME
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_cifar_resnet/$TIME
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR

python ${CONFIG_DIR}/${ANN_APP_DIR}/resnet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
-n=25 -c=32 -s=3 \
--ds_name=cifar100 --batch_size=64 --nr_gpu=1 --samloss=100 --adaloss_gamma=0.07 --adaloss_momentum=0.99 --adaloss_final_extra=1.0 --adaloss_update_per=100 --sum_rand_ratio=0 --is_select_arr -f=5 --resnet_version=resnext
