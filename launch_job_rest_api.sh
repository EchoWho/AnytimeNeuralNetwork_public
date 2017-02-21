#!/bin/bash

USERNAME="dedey"
PASSWORD="Yoga2god?"
CLUSTER="gcr"
JOBSCRIPT="run_exp_141.sh"
SPECIAL_NAME="_ann"

VC="pnrsy"
CMD="https://philly/api/submit?"
CMD+="buildId=0000&"
CMD+="customDockerName=custom-tf-0-12-python-2-7-ver2&"
CMD+="toolType=cust&"
CMD+="clusterId=$CLUSTER&"
CMD+="vcId=$VC&"
CMD+="configFile=$USERNAME%2FAnytimeNeuralNetwork%2F$JOBSCRIPT&"
CMD+="minGPUs=1&"
CMD+="name=cust-p-$JOBSCRIPT$SPECIAL_NAME!~!~!1&"
CMD+="isdebug=false&"
CMD+="iscrossrack=false&"
CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2F&"
CMD+="userName=$USERNAME"

curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"