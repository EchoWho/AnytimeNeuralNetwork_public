#!/bin/bash
ds_name=camvid
export CUDA_VISIBLE_DEVICES=2,3
DATA_DIR=${GLOBAL_DATA_DIR}
LOG_DIR=${GLOBAL_LOG_DIR}/${ds_name}_ann
MODEL_DIR=${GLOBAL_MODEL_DIR}/${ds_name}_ann
CONFIG_DIR=.

mkdir -p $MODEL_DIR

python $CONFIG_DIR/anytime_models/examples/fcdense-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--ds_name=${ds_name} \
--densenet_version=atv1 \
-f=5 \
--opt_at=-1 \
--n_blocks=11 \
--n_pools=5 \
--fcdense_depth=103 \
-s=1 \
--batch_size=6 \
--nr_gpu=2 \
--init_lr=1e-3 \
--use_init_ch \
-c=72 \
-g=24 \
--regularize_const=1e-4 \
--is_label_one_hot \
--dense_select_method=0 \
--early_connect_type=1 \
--log_dense_coef=2.0 \
--weights_at_block_ends \
--operation=evaluate \
--is_test \
--load=${MODEL_DIR}/model-43432.index \

# for finetune
#--load=${MODEL_DIR}/model-42517.index \
# for eval
#--load=${MODEL_DIR}/model-43432.index \
# whether evluate on test or validation
#--is_test \
