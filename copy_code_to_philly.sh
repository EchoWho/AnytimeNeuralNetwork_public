#!/bin/bash

# Mount the msrlabs_scratch/dedey folder to /mnt/msrlabs_scratch_dedey
sudo mount -t cifs //storage.cam.philly.selfhost.corp.microsoft.com/msrlabs_scratch/dedey /mnt/msrlabs_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8

# Copy AnytimeNeuralNetwork contents to /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork
sudo cp -r /home/dedey/AnytimeNeuralNetwork/* /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master

# Unmount the /mnt/pnrsy_scratch_dedey folder
sudo umount /mnt/msrlabs_scratch_dedey