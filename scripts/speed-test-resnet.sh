#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# Note that $s need to be adjusted as well. (d, s):
# (14, 1), (26, 2), (50, 2), (101, 3), (18, 2), (34, 2), (152, 5)

for dns in 14,1,4 26,2,4 50,2,8 101,3,11
do
	d=`echo $dns | grep "^[0-9]*," -o | grep "[0-9]*" -o`
	s=`echo $dns | grep ",[0-9]*," -o | grep "[0-9]*" -o`
	max_np=`echo $dns | grep ",[0-9]*$" -o | grep "[0-9]*" -o`

	for np in `seq ${max_np}`
	do

		DATA_DIR=${GLOBAL_DATA_DIR}
		model_dn=imagenet-resnet-ann-d${d}-s${s}-np${np}
		LOG_DIR=${GLOBAL_LOG_DIR}/resnet-speed-test/${model_dn}
		MODEL_DIR=${GLOBAL_MODEL_DIR}/imagenet_model/${model_dn}
		CONFIG_DIR=.

		mkdir -p $MODEL_DIR

		echo $model_dn

		# Run the actual job
		python $CONFIG_DIR/anytime_models/examples/imagenet-ann.py \
		--data_dir=$DATA_DIR \
		--log_dir=$LOG_DIR \
		--model_dir=$MODEL_DIR \
		-d=${d} -s=${s} --num_anytime_preds=${np} \
		--batch_size=64 --nr_gpu=1 \
		-f=5 --samloss=100 --exp_gamma=0.07 --sum_rand_ratio=0 --is_select_arr \
		--opt_at=-1 -c=64 --num_classes=1000 \
		--evaluate=val
	done
done
