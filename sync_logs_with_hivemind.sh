#!/bin/bash

rsync -r -v -e ssh debadeepta@hivemind.ml.cmu.edu:/data/hanzhang/train_log_dey/ann_models_logs/ ann_models_logs/
