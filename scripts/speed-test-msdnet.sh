#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

for trial in `seq 5`
do
for dns in 38,1000,5
do
	d=`echo $dns | grep "^[0-9]*," -o | grep "[0-9]*" -o`
	s=`echo $dns | grep ",[0-9]*," -o | grep "[0-9]*" -o`
	max_np=`echo $dns | grep ",[0-9]*$" -o | grep "[0-9]*" -o`

	for np in `seq ${max_np}`
	do

		DATA_DIR=${GLOBAL_DATA_DIR}
		model_dn=imagenet-msdnet-d${d}-np${np}-trial${trial}
		LOG_DIR=${GLOBAL_LOG_DIR}/msdnet-speed-test/${model_dn}
		MODEL_DIR=${GLOBAL_MODEL_DIR}/imagenet_model/${model_dn}
		CONFIG_DIR=.

		mkdir -p $MODEL_DIR

		echo $model_dn

		# Run the actual job
		python $CONFIG_DIR/anytime_models/examples/imagenet-msdense-ann.py \
		--data_dir=$DATA_DIR \
		--log_dir=$LOG_DIR \
		--model_dir=$MODEL_DIR \
		--num_anytime_preds=${np} \
		-f=5 --samloss=100 --exp_gamma=0.07 --sum_rand_ratio=0 --is_select_arr  \
		--msdensenet_depth=${d} \
		--batch_size=64 --nr_gpu=1 --opt_at=-1 -g=16 --num_classes=1000 --num_scales=4 --prediction_feature=msdense --min_predict_unit=4 \
		--evaluate=val
	done
done
done
