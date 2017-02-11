#!/bin/bash

# Mount the pnrsy_scratch/dedey folder to /mnt/pnrsy_scratch_dedey
sudo mount -t cifs //storage.gcr.philly.selfhost.corp.microsoft.com/pnrsy_scratch/dedey /mnt/pnrsy_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8

# Copy AnytimeNeuralNetwork contents to /mnt/pnrsy_scratch_dedey/AnytimeNeuralNetwork
sudo cp -r /home/dedey/AnytimeNeuralNetwork/* /mnt/pnrsy_scratch_dedey/AnytimeNeuralNetwork

# Unmount the /mnt/pnrsy_scratch_dedey folder
sudo umount /mnt/pnrsy_scratch_dedey