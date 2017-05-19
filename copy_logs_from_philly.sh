#!/bin/bash

#---------------CHANGE BELOW---------------------------
APPNAME="application_1493880963148_3375"
CLUSTER="gcr"
VC="msrlabs_scratch"
COPYLOC="/home/dedey/DATADRIVE2/ann_models_logs/delete"
#----------DO NOT CHANGE BELOW-------------------------


FROM_LOC="//storage.$CLUSTER.philly.selfhost.corp.microsoft.com/$VC/sys/jobs/$APPNAME"
TO_LOC="/mnt/experiment_log"

# Mount the etc folder for the job from philly to local machine
echo $FROM_LOC $TO_LOC
sudo mount -t cifs $FROM_LOC $TO_LOC -o username=dedey,domain=REDMOND,iocharset=utf8

# Make directory to copy stuff to if not exists
mkdir -p $COPYLOC

# Copy
cp -r "$TO_LOC/"* "$COPYLOC/"

# Unmount
sudo umount $TO_LOC

