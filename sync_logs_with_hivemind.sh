#!/bin/bash

rsync -r -v -e ssh /home/dedey/DATADRIVE1/ann_models_logs/ debadeepta@hivemind.ml.cmu.edu:/data2/saved_models_ann/ann_models_logs/
