#!/bin/bash

# This script parses in the command line parameters from runCust, 
# maps them to the correct command line parameters for DispNet training script and launches that task
# The last line of runCust should be: bash $CONFIG_FILE --data-dir $DATA_DIR --log-dir $LOG_DIR

# Parse the command line parameters
# that runCust will give out

DATA_DIR=NONE
LOG_DIR=NONE
CONFIG_DIR=NONE
MODEL_DIR=NONE

# Parsing command line arguments:
while [[ $# > 0 ]]
do
key="$1"

case $key in
    -h|--help)
    echo "Usage: run_dispnet_training_philly.sh [run_options]"
    echo "Options:"
    echo "  -d|--data-dir <path> - directory path to input data (default NONE)"
    echo "  -l|--log-dir <path> - directory path to save the log files (default NONE)"
    echo "  -p|--config-file-dir <path> - directory path to config file directory (default NONE)"
    echo "  -m|--model-dir <path> - directory path to output model file (default NONE)"
    exit 1
    ;;
    -d|--data-dir)
    DATA_DIR="$2"
    shift # pass argument
    ;;
    -p|--config-file-dir)
    CONFIG_DIR=`dirname $2`
    shift # pass argument
    ;;
    -m|--model-dir)
    MODEL_DIR="$2"
    shift # pass argument
    ;;
    -l|--log-dir)
    LOG_DIR="$2"
    shift
  ;;
    *)
    echo Unkown option $key
    ;;
esac
shift # past argument or value
done

# Prints out the arguments that were passed into the script
echo "DATA_DIR=$DATA_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "CONFIG_DIR=$CONFIG_DIR"
echo "MODEL_DIR=$MODEL_DIR"

# Run training on philly

# Add the root folder of the code to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CONFIG_DIR

# Run the actual job
python $CONFIG_DIR/anytime_models/examples/imagenet-msdense-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--load=${MODEL_DIR}/checkpoint \
-f=5 --samloss=100  --block_config=8,8,8,9 -s=8 --batch_size=256 --nr_gpu=4 --min_predict_unit=8 --adaloss_gamma=0.1 --adaloss_final_extra=3 --adaloss_update_per=1000 --opt_at=-1 --num_classes=1000 
