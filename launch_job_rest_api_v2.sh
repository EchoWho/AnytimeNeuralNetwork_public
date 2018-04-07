#!/bin/bash


USERNAME="dedey"
CLUSTER="gcr"
JOBSCRIPT="run_exp_3892.sh"
SPECIAL_NAME="_ann"
VC="msrlabs"
NUM_GPUS="4"

curl -H "Content-Type: application/json" \
     -H "WWW-Authenticate: Negotiate" \
     -H "WWW-Authenticate: NTLM" \
     -X POST https://philly/api/v2/submit -k --ntlm -n -d \
     '{
        "ClusterId": "gcr",
        "VcId": "msrlabs",
        "JobName": "cust-p-run_exp_3892_ann",
        "UserName": "dedey",
        "BuildId": 0,
        "ToolType": null,
        "ConfigFile": "dedey/AnytimeNeuralNetwork_master/run_exp_3892.sh",
        "Inputs": [{
            "Name": "dataDir",
            "Path": "/hdfs/msrlabs/dedey/ann_data_dir"
        }],
        "Outputs": [],
        "IsDebug": false,
        "CustomDockerName": "doesnotmatter",
        "RackId": "anyConnected",
        "MinGPUs": 4,
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




# generate_post_data()
# {
#   cat <<EOF
# {
#   "account": {
#     "email": "$email",
#     "screenName": "$screenName",
#     "type": "$theType",
#     "passwordSettings": {
#       "password": "$password",
#       "passwordConfirm": "$password"
#     }
#   },
#   "firstName": "$firstName",
#   "lastName": "$lastName",
#   "middleName": "$middleName",
#   "locale": "$locale",
#   "registrationSiteId": "$registrationSiteId",
#   "receiveEmail": "$receiveEmail",
#   "dateOfBirth": "$dob",
#   "mobileNumber": "$mobileNumber",
#   "gender": "$gender",
#   "fuelActivationDate": "$fuelActivationDate",
#   "postalCode": "$postalCode",
#   "country": "$country",
#   "city": "$city",
#   "state": "$state",
#   "bio": "$bio",
#   "jpFirstNameKana": "$jpFirstNameKana",
#   "jpLastNameKana": "$jpLastNameKana",
#   "height": "$height",
#   "weight": "$weight",
#   "distanceUnit": "MILES",
#   "weightUnit": "POUNDS",
#   "heightUnit": "FT/INCHES"
# }
# EOF 