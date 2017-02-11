#!/bin/bash

USERNAME="dedey"
PASSWORD="Yoga2god?"
CLUSTER="gcr"
JOBSCRIPT="AnytimeNeuralNetwork/run_exp_2.sh"

VC="pnrsy"
CMD="https://philly/api/submit?"
CMD+="buildId=0000&"
CMD+="customDockerName=custom-tf-0-12-python-2-7&"
CMD+="toolType=cust&"
CMD+="clusterId=$CLUSTER&"
CMD+="vcId=$VC&"
CMD+="configFile=$USERNAME%2F$JOBSCRIPT&"
CMD+="minGPUs=1&"
CMD+="name=cust-p-$JOBSCRIPT!~!~!1&"
CMD+="isdebug=false&"
CMD+="iscrossrack=false&"
CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2F&"
CMD+="userName=$USERNAME"

curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"