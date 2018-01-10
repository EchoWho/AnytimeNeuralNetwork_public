#!/bin/bash
# Exp 67 (imagenet resnet 50, 101)
# 3118..3119
# Num gpus: 4

# Exp 68 (imagenet msdense 23, 33, 38)
# 3122..3124
# Num gpus: 4

# Exp 69 (imagenet msdense 23, 33, 38, nr_gpu=8
# 3127..3129
# Num gpus: 8

for i in {3127..3129}
do
	USERNAME="dedey"
	PASSWORD="DigDug2god?"
	CLUSTER="cam"
	JOBSCRIPT="run_exp_$i.sh"
	SPECIAL_NAME="_ann"
	VC="msrlabs"
	NUM_GPUS="8"

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

# Exp 46:
# 2327..2428
# Num gpus: 1

# Exp 47:
# 2431..2450
# Num gpus: 1

# Exp 48:
# 2453..2503
# Num gpus: 2

# Exp 49:
# 2506..2556
# Num gpus: 1

# Exp 50:
# 2559..2561
# Num gpus: 4

# Exp 51:
# 2564..2587
# Num gpus: 2

# Exp 52:
# 2590..2640
# Num gpus: 2

# Exp 53:
# 2643..2666
# Num gpus: 2

# Exp 54:
# 2669..2692
# Num gpus: 2

# Exp 55:
# 2695..2718
# Num gpus: 2

# Exp 56:
# 2721..2744
# Num gpus: 2

# Exp 57:
# 2747..2770
# Num gpus: 2

# Exp 58:
# 2773..2796
# Num gpus: 2

# Exp 59:
# 2799..2822
# Num gpus: 2

# Exp 60
# 2825..2878
# Num gpus: 4

# Exp 61
# 2881..2901
# Num gpus: 4

# Exp 62
# 2904..2957
# Num gpus: 4

# Exp 63
# 2960..2980
# Num gpus: 4

# Exp 64
# 2983..3009
# Num gpus: 4

# Exp 65
# 3011..3032
# Num gpus: 4

# Exp 66
# 3035..3115
# Num gpus: 4

# Exp 67 (imagenet resnet 50, 101)
# 3118..3119
# Num gpus: 4

# Exp 68 (imagenet msdense 23, 33, 38)
# 3122..3124
# Num gpus: 4

# Exp 69 (imagenet msdense 23, 33, 38, nr_gpu=8
# 3127..3129
# Num gpus: 8
