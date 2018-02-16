#!/bin/bash
# Exp 80 (cifar/svhn adaloss competitor)
# 3351..3446
# Num gpus : 1

# Exp 81 (resnet ilsvrc adaloss 4gpu)
# 3449..3452
# Num gpus : 4

# Exp 82 (resnet ilsrvc adaloss 8gpu copy of 81)
# 3455..3458
# Num gpus : 8

# Exp 83 (msdense ilsvrc adaloss 4gpu)
# 3461..3469
# Num gpus : 4

# Exp 84 (msdense copy of 83 w/ 8gpu)
# 3472..3480
# Num gpus : 8

# Exp 85 (cifar/svhn adaloss with s=1)
# 3483..3518
# Num gpus : 1

# Exp 86 (DenseNet reproduce round 3)
# 3521..3523
# Num gpus : 4

# Exp 87 (copy of 86 with 8 gpu)
# 3526..3528
# Num gpus : 8

# FOR I IN {3449..3452} {3461..3469} {3521..3523} # 4GPU ;   
# do
# 	USERNAME="dedey"
# 	PASSWORD="DigDug2god?"
# 	CLUSTER="cam"
# 	JOBSCRIPT="run_exp_$i.sh"
# 	SPECIAL_NAME="_ann"
# 	VC="msrlabs"
# 	NUM_GPUS="4"

# 	CMD="https://philly/api/submit?"
# 	CMD+="buildId=0000&"
# 	CMD+="customDockerName=custom-tf-1-1-0-python-2-7&"
# 	CMD+="toolType=cust&"
# 	CMD+="clusterId=$CLUSTER&"
# 	CMD+="vcId=$VC&"
# 	CMD+="configFile=$USERNAME%2FAnytimeNeuralNetwork_master%2F$JOBSCRIPT&"
# 	CMD+="minGPUs=$NUM_GPUS&"
# 	CMD+="name=cust-p-$JOBSCRIPT$SPECIAL_NAME!~!~!1&"
# 	CMD+="isdebug=false&"
# 	CMD+="iscrossrack=false&"
# 	CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2F&"
# 	CMD+="oneProcessPerContainer=true&"
# 	CMD+="userName=$USERNAME"

# 	curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"

# 	echo "$CMD"
# done

# For running jobs on PhillyOnAzure

for i in {3455..3458} {3472..3480} {3526..3528} #8gpu
do
	USERNAME="dedey"
	PASSWORD="DigDug2god?"
	CLUSTER="eu1"
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
	CMD+="rackid=p100-gpc02&"
	CMD+="userName=$USERNAME"

	echo "$CMD"

	curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"

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

# Exp 70 (imagenet msdense f=4 for control)
# 3132..3134
# Num gpus: 8

# Exp 71 (imagenet ResNeXt nr_gpu=8)
# 3137..3144
# Num gpus: 8

# Exp 72 (imagenet DenseNet nr_gpu=8)
# 3147..3154
# Num gpus: 8

# Exp 73 (cifar svhn with adaloss)
# 3157..3204
# Num gpus : 1

# Exp 74 (cifar svhn with resnext)
# 3207..3296
# Num gpus : 4

# Exp 75 (imagenet DenseNet reproduce rerun)
# 3299..3306 
# Num gpus : 8

# Exp 76 (resnext f=4)
# 3309..3326
# Num gpus : 4

# Exp 77 (resnext imagenet)
# 3329..3332
# Num gpus : 8

# Exp 78 (resnext imagenet gpu 4)
# 3335..3338
# Num gpus : 4

# Exp 79 (densenet reproduce with 4 gpu)
# 3341..3348
# Num gpus : 4

# Exp 80 (cifar/svhn adaloss competitor)
# 3351..3446
# Num gpus : 1

# Exp 81 (resnet ilsvrc adaloss 4gpu)
# 3449..3452
# Num gpus : 4

# Exp 82 (resnet ilsrvc adaloss 8gpu copy of 81)
# 3455..3458
# Num gpus : 8

# Exp 83 (msdense ilsvrc adaloss 4gpu)
# 3461..3469
# Num gpus : 4

# Exp 84 (msdense copy of 83 w/ 8gpu)
# 3472..3480
# Num gpus : 8

# Exp 85 (cifar/svhn adaloss with s=1)
# 3483..3518
# Num gpus : 1

# Exp 86 (DenseNet reproduce round 3)
# 3521..3523
# Num gpus : 4

# Exp 87 (copy of 86 with 8 gpu)
# 3526..3528
# Num gpus : 8
