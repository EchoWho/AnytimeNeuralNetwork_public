import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils import utils

from tensorpack.network_models import anytime_network
from tensorpack.network_models.anytime_network import AnytimeFCDensenet

"""
"""
INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None

def get_camvid_data(which_set, shuffle=True):
    isTrain = which_set == 'train' or which_set == 'trainval'

    pp_mean = dataset.Camvid.mean
    pp_std = dataset.Camvid.std
    
    ds = dataset.Camvid(which_set, shuffle)
    ds = BatchData(ds, args.batch_size // args.nr_gpu, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds



def get_config(ds_trian, ds_val, model_cls):
    # prepare dataset
    steps_per_epoch = ds_train.size() // args.nr_gpu

    model=model_cls(args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, keep_freq=12),
            InferenceRunner(ds_val,
                            [ScalarStats('cost')] + classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate')
        ] + loss_select_cbs,
        model=model,
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset choice 
    parser.add_argument('--ds_name', help='name of dataset',
                        type=str, 
                        choices=['camvid'])
    # other common args
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=3)
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
    parser.add_argument('--model_dir', help='model_dir position',
                        type=str, default=None)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--do_validation', help='Whether use validation set. Default not',
                        type=bool, default=False)
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--is_toy', help='Whether to have data size of only 1024',
                        type=bool, default=False)
    parser.add_argument('--is_test', help='Whehter use train-val or trainval-test',
                        type=bool, default=False)
    anytime_network.parser_add_fcdense_arguments(parser)
    model_cls = AnytimeFCDensenet
    args = parser.parse_args()

    ## Set dataset-network specific assert/info
    #
    # Make sure the input images have H/W that are divisible by
    # 2**n_pools; see tensorpack/network_models/anytime_network.py
    if args.ds_name == 'camvid':
        args.num_classes = 12
        INPUT_SIZE = None
        get_data = get_camvid_data
        if not args.is_test:
            ds_train = get_data('train') #trainval
            ds_val = get_data('val') #test
        else:
            ds_train = get_data('trainval')
            ds_val = get_data('test')

        lr_schedule = \
            [(1, 0.01), (15, 1e-3), (30, 1e-4), (45, 1e-5)]
        max_epoch = 75

    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))

    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()

    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(arg.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
