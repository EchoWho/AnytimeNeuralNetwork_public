#!/bin/bash

# Exp 90 (LogDense 4 gpu)
# 3553..3558
# Num gpus : 4

# Exp 91 (LogDense 8 gpu)
# 3561..3566
# Num gpus : 8

# Exp 92 (adaloss grid search on cifar10, cifar100)
# 3569..3784
# Num gpus : 1

# Exp 93 (baseline optimal on cifar 100 for n=9, n=17, n=25
# 3787..3889
# Num gpus : 1

# Exp 94 (DenseNet reproduction with dropout =0.9)
# 3892..3900
# Num gpus : 4

# Exp 95 (Copy of 94 on 8 gpus)
# 3903..3911
# Num gpus : 8

# Exp 96 (ResNext with adaloss)
# 3914..3925
# Num gpus : 4

# Exp 97 (Copy of 96 w/ 8 gpus)
# 3928..3939
# Num gpus : 8

# Exp 98 (ResNet with adaloss with grid searched param)
# 3942..3945
# Num gpus : 4

# Exp 99 (Copy of 98 w/ 8 gpus)
# 3948..3951
# Num gpus : 8

# Exp 100 (MSDNet with adaloss with grid searched param)
# 3954..3956
# Num gpus : 4

# Exp 101 (Copy of 100 w/ 8 gpus)
# 3959..3961
# Num gpus : 8


# Exp 102 (1 GPU and 2 GPU benchmark)
# 3964..3965
# Num gpus : varies {3964 : 1, 3965 : 2}

# Exp 103 (2 GPU msdnet cifar (done on cmu machines)
# 3968..3973
# Num gpus : 2

# Exp 104 (4 gpu dense with const scheme)
# 3976..3979
# Num gpus : 4

# Exp 105 (8 gpu copy of 104)
# 3982..3985
# Num gpus : 8

# Exp 106 (Small DenseNet)
# 3988..3993
# Num gpus : 4

# Exp 107 (DenseNet with rescale prediction features)
# 3996..4007
# Num gpus : 4

# Exp 108 (8 gpu copy of 107)
# 4010..4021
# Num gpus : 8

# Exp 109 (1 gpu cifar/svhn rescale feature)
# 4024..4068
# Num gpus : 1

# Exp 110 (4 gpu resnet with smaller number of prediction and better anytime predictors)
# 4071..4078
# Num gpus : 4

# Exp 111 (4 GPU small msdense for adaloss vs const story)
# 4081..4085
# Num gpus : 4

# Exp 112 (4 GPU small msdnet with param sweep)
# 4088..4107
# Num gpus : 4

# Exp 113 (4 GPU MSDNet 30 and MSDNet38)
# 4110..4115
# Num gpus : 4

# Exp 114 (4 GPU DenseNet)
# 4118..4126
# Num gpus : 4

# Exp 115 (4 GPU MSDNet const baselin)
# 4129..4130
# Num gpus : 4

for i in {4110..4115} {4118..4126} {4129..4130}
do
    USERNAME="dedey"
    CLUSTER="eu1"
    JOBSCRIPT="run_exp_$i.sh"
    SPECIAL_NAME="_ann"
    VC="msrlabs"
    NUM_GPUS="4"

    curl -H "Content-Type: application/json" \
         -H "WWW-Authenticate: Negotiate" \
         -H "WWW-Authenticate: NTLM" \
         -X POST https://philly/api/v2/submit -k --ntlm -n -d \
         '{
            "ClusterId": "'${CLUSTER}'",
            "VcId": "'${VC}'",
            "JobName": "cust-p-'${JOBSCRIPT}''${SPECIAL_NAME}'",
            "UserName": "'${USERNAME}'",
            "BuildId": 0,
            "ToolType": null,
            "ConfigFile": "/blob/'${VC}'/'${USERNAME}'/AnytimeNeuralNetwork_master/cust_exps/'${JOBSCRIPT}'",
            "Inputs": [{
                "Name": "dataDir",
                "Path": "/hdfs/'${VC}'/'${USERNAME}'/ann_data_dir"
            }],
            "Outputs": [],
            "IsDebug": false,
            "CustomDockerName": "doesnotmatter",
            "RackId": "anyConnected",
            "MinGPUs": '${NUM_GPUS}',
            "PrevModelPath": null,
            "ExtraParams": null,
            "SubmitCode": "p",
            "IsMemCheck": false,
            "IsCrossRack": false,
            "Registry": "phillyregistry.azurecr.io",
            "RegistryUsername": null,
            "RegistryPassword": null,
            "Repository": "philly/jobs/custom/tensorflow",
            "Tag": "tf16-py27",
            "OneProcessPerContainer": true
        }'
echo
done



