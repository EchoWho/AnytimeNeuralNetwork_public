#!/bin/bash

for i in {1132..1455}
do

	USERNAME="dedey"
	PASSWORD="Chakra2god?"
	CLUSTER="gcr"
	JOBSCRIPT="run_exp_$i.sh"
	SPECIAL_NAME="_ann"
	VC="msrlabs"
	NUM_GPUS="1"

	CMD="https://philly/api/submit?"
	CMD+="buildId=0000&"
	CMD+="customDockerName=custom-tf-1-1-python-2-7&"
	CMD+="toolType=cust&"
	CMD+="clusterId=$CLUSTER&"
	CMD+="vcId=$VC&"
	CMD+="configFile=$USERNAME%2FAnytimeNeuralNetwork_master%2F$JOBSCRIPT&"
	CMD+="minGPUs=$NUM_GPUS&"
	CMD+="name=cust-p-$JOBSCRIPT$SPECIAL_NAME!~!~!1&"
	CMD+="isdebug=false&"
	CMD+="iscrossrack=false&"
	CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2F&"
	CMD+="oneProcessPerContainer=true&"
	CMD+="userName=$USERNAME"

	curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"

	echo "$CMD"

	# FOR WHEN YOU NEED IMAGENET	
	# CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2Fimagenet_tfrecords%2F&"

	# FOR WHEN YOU NEED OTHER DATASETS
	# CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2F&"

	# Test here

done
