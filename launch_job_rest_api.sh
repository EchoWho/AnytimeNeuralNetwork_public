#!/bin/bash

#for i in {1638..1685} {1688..1711} {1714..1719} {1768..1803}
for i in {2301..2324} {2248..2298}
do

	USERNAME="dedey"
	PASSWORD="Valar2god?"
	CLUSTER="gcr"
	JOBSCRIPT="run_exp_$i.sh"
	SPECIAL_NAME="_ann"
	VC="msrlabs"
	NUM_GPUS="1"

	CMD="https://philly/api/submit?"
	CMD+="buildId=0000&"
	CMD+="customDockerName=custom-tf-1-1-0-python-2-7&"
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

done


# Exp 31
# 1638..1685
# Num gpus: 1

# Exp 32
# 1688..1711
# Num gpus: 1

# Exp 33
# 1714..1719
# Num gpus: 1
# 1720..1722
# Num gpus: 2

# Exp 35
# 1737..1745
# Num gpus: 1
# 1743..1745
# Num gpus: 2

# Exp 36
# 1748..1765
# Num gpus: 1

# Exp 37
# 1768..1803
# Num gpus: 1

# Exp 39:
# 1812..1889
# Num gpus: 1

# Exp 40:
# 1892..1969
# Num gpus: 1

# Exp 41:
# 1972..2022
# Num gpus: 1

# Exp 42:
# 2025..2141
# Num gpus: 1

# Exp 43:
# 2144..2245
# Num gpus: 1


# Exp 44:
# 2248..2298
# Num gpus: 1

# Exp 45:
# 2301..2324
# Num gpus: 1
