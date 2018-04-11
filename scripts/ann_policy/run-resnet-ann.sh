#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
n=$1
s=$(((${n} * 3 + 4)/ 8))

echo $n $s
c=16

DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/ann_policy/cifar100/n${n}-c${c}-s${s}
MODEL_DIR=${GLOBAL_MODEL_DIR}/ann_policy/cifar100/n${n}-c${c}-s${s}
CONFIG_DIR=.

mkdir -p $MODEL_DIR

python $CONFIG_DIR/examples/AnytimeNetwork/resnet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--ds_name=cifar100 \
-f=10 \
--opt_at=-1 \
-n=${n} \
-c=${c} \
-s=${s} \
--samloss=6 \
--batch_size=64 \
--do_validation \
