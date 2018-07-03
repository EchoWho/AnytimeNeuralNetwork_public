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
    CONFIG_DIR="$2"
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
python $CONFIG_DIR/examples/AnytimeNetwork/densenet-ann.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--load=${MODEL_DIR}/checkpoint \
--densenet_version=atv1 -n=82 -g=32 -s=41 --ds_name=cifar100 --batch_size=32 --nr_gpu=2 --reduction_ratio=1 --dropout_kp=1 -f=2 --opt_at=-1 --samloss=0 --log_dense_base=2.0 --transition_batch_size=1 --growth_rate_multiplier=1 --bottleneck_width=4.0 --b_type=bottleneck 
