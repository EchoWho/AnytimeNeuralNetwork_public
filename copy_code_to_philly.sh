#!/bin/bash

echo "Copying code to gcr"

# Mount the msrlabs_scratch/dedey folder to /mnt/msrlabs_scratch_dedey
sudo mount -t cifs //storage.gcr.philly.selfhost.corp.microsoft.com/msrlabs_scratch/dedey /mnt/msrlabs_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8,passwd="Urdu2god?"

# Copy AnytimeNeuralNetwork contents to /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork
sudo rsync -vr --progress --exclude='.git/' /home/dedey/AnytimeNeuralNetwork/ /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/

# Unmount the /mnt/msrlabs_scratch_dedey folder
sudo umount /mnt/msrlabs_scratch_dedey

echo "Copying code to rr1"

# Mount the msrlabs_scratch/dedey folder to /mnt/msrlabs_scratch_dedey
sudo mount -t cifs //storage.rr1.philly.selfhost.corp.microsoft.com/msrlabs_scratch/dedey /mnt/msrlabs_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8,passwd="Urdu2god?"

# Copy AnytimeNeuralNetwork contents to /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork
sudo rsync -vr --progress --exclude='.git/' /home/dedey/AnytimeNeuralNetwork/ /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/

# Unmount the /mnt/msrlabs_scratch_dedey folder
sudo umount /mnt/msrlabs_scratch_dedey


echo "Copying code to cam"

# Mount the msrlabs_scratch/dedey folder to /mnt/msrlabs_scratch_dedey
sudo mount -t cifs //storage.cam.philly.selfhost.corp.microsoft.com/msrlabs_scratch/dedey /mnt/msrlabs_scratch_dedey/ -o username=dedey,domain=REDMOND,iocharset=utf8,passwd="Urdu2god?"

# Copy AnytimeNeuralNetwork contents to /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork
sudo rsync -vr --progress --exclude='.git/' /home/dedey/AnytimeNeuralNetwork/ /mnt/msrlabs_scratch_dedey/AnytimeNeuralNetwork_master/

# Unmount the /mnt/msrlabs_scratch_dedey folder
sudo umount /mnt/msrlabs_scratch_dedey
