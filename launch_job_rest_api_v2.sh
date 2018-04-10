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

for i in {3942..3945} {3954..3956} # 4 gpu ;   {3948..3951} {3959..3961} # 8 gpu
do
    USERNAME="dedey"
    CLUSTER="cam"
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
            "ConfigFile": "'${USERNAME}'/AnytimeNeuralNetwork_master/cust_exps/'${JOBSCRIPT}'",
            "Inputs": [{
                "Name": "dataDir",
                "Path": "/hdfs/'${VC}'/dedey/ann_data_dir"
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
done



