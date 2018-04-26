#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_cifar_msdnet/3970
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_cifar_msdnet/3970
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR

python ${CONFIG_DIR}/${ANN_APP_DIR}/msdensenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
-f=5 --samloss=100  --msdensenet_depth=27 -s=4 --batch_size=64 --nr_gpu=1 --ds_name=cifar100 --opt_at=-1
