#!/bin/bash
ds_name=cifar100  #cifar10, cifar100 or svhn
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_DIR=${GLOBAL_DATA_DIR}/
LOG_DIR=${GLOBAL_LOG_DIR}/${ds_name}/
MODEL_DIR=${GLOBAL_MODEL_DIR}/${ds_name}/
CONFIG_DIR=.

mkdir -p $MODEL_DIR

python $CONFIG_DIR/examples/AnytimeNetwork/resnet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--ds_name=${ds_name} \
-f=5 \
-n=5 \
-c=16 \
--samloss=6 \
--batch_size=64 \
#--alter_label \
#--alter_label_activate_frac=0.75 \
#--alter_loss_w=1.0 \
