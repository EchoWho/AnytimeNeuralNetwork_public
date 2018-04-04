#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
TIME=`date +"%y_%m_%d_%H_%M"`
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/test_cifar_msdnet/$TIME
MODEL_DIR=${GLOBAL_MODEL_DIR}/test_cifar_msdnet/$TIME
CONFIG_DIR=.
ANN_APP_DIR=anytime_models/examples

mkdir -p $MODEL_DIR
