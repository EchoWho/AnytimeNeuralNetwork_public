#!/bin/bash


sudo find /home/dedey/ann_models_logs/  -name events.out.tfevents*  -exec rm -f {} \;
rsync -r -v -e ssh /home/dedey/ann_models_logs/ debadeepta@hivemind.ml.cmu.edu:/data2/saved_models_ann/ann_models_logs/
