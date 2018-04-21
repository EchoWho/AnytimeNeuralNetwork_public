#!/bin/bash

# echo "Copying code to gcr"

# # Mount the msrlabs_scratch/dedey folder to /mnt/msrlabs_scratch_dedey
# sudo mount -t cifs //storage.gcr.philly.selfhost.corp.microsoft.com/msrlabs_scratch/dedey /mnt/msrlabs_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8,passwd="Krishna2god?"

# # Clean
# sudo rm -rf /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/* 

# # Copy AnytimeNeuralNetwork contents to /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork
# sudo rsync -vr --progress --exclude='.git/' /home/dedey/AnytimeNeuralNetwork/ /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/

# # Unmount the /mnt/msrlabs_scratch_dedey folder
# sudo umount /mnt/msrlabs_scratch_dedey

# echo "Copying code to rr1"

# # Mount the msrlabs_scratch/dedey folder to /mnt/msrlabs_scratch_dedey
# sudo mount -t cifs //storage.rr1.philly.selfhost.corp.microsoft.com/msrlabs_scratch/dedey /mnt/msrlabs_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8,passwd="Krishna2god?"

# # Clean
# sudo rm -rf /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/* 

# # Copy AnytimeNeuralNetwork contents to /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork
# sudo rsync -vr --progress --exclude='.git/' /home/dedey/AnytimeNeuralNetwork/ /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/

# # Unmount the /mnt/msrlabs_scratch_dedey folder
# sudo umount /mnt/msrlabs_scratch_dedey


echo "Copying code to cam"

# Mount the msrlabs_scratch/dedey folder to /mnt/msrlabs_scratch_dedey
sudo mount -t cifs //storage.cam.philly.selfhost.corp.microsoft.com/msrlabs_scratch/dedey /mnt/msrlabs_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8,passwd="Krishna2god?"

# Basically create a zip
mkdir -p /home/dedey/AnytimeNeuralNetwork_master
cp -r /home/dedey/AnytimeNeuralNetwork/* /home/dedey/AnytimeNeuralNetwork_master/
tar -cvzf /home/dedey/AnytimeNeuralNetwork_master.tar.gz -C /home/dedey/AnytimeNeuralNetwork_master .
sudo cp /home/dedey/AnytimeNeuralNetwork_master.tar.gz /mnt/msrlabs_scratch_dedey/ 
sudo tar -xvzf /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master.tar.gz --no-same-owner --skip-old-files -C /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/
rm -r /home/dedey/AnytimeNeuralNetwork_master /home/dedey/AnytimeNeuralNetwork_master.tar.gz

# Unmount the /mnt/msrlabs_scratch_dedey folder
sudo umount /mnt/msrlabs_scratch_dedey
