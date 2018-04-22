#!/bin/bash

COPY_LOC=$1
FROM_LOC=$2

TO_LOC="/mnt/experiment_log"

# Mount the etc folder for the job from philly to local machine
echo $FROM_LOC $TO_LOC
sudo mount -t cifs $FROM_LOC $TO_LOC -o username=dedey,password="Krishna2god?",domain=REDMOND,iocharset=utf8

# Make directory to copy stuff to if not exists
mkdir -p $COPY_LOC

# Copy
sudo cp -r "$TO_LOC/"* "$COPY_LOC/"

# Unmount
#sudo fuser -kim $TO_LOC  # kill any processes accessing file
sudo umount -fl $TO_LOC
