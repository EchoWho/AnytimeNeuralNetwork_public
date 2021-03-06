#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_cifar_msdnet/3968
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_cifar_msdnet/3968
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR

python ${CONFIG_DIR}/${ANN_APP_DIR}/msdensenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--ds_name=cifar100 \
-f=5 --samloss=100  --msdensenet_depth=18 -s=2 --batch_size=64 --nr_gpu=1 --opt_at=-1 --min_predict_unit=4
