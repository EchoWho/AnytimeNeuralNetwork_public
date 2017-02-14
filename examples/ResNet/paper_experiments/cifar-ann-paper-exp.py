#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import argparse
import os

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
import tensorpack.utils.anytime_loss as anytime_loss
from tensorpack.utils import logger

from tensorflow.contrib.layers import variance_scaling_initializer

"""
"""
import imp
dir_name = os.path.dirname(__file__)
cifar_example = imp.load_source('cifar_example',
                                os.path.join(dir_name[:dir_name.rfind('/')], 'cifar-ann.py'))
Model = cifar_example.Model

BATCH_SIZE = 128
NUM_RES_BLOCKS = 3
NUM_UNITS = 5
WIDTH = 1
INIT_CHANNEL = 16
NUM_CLASSES = 10

FUNC_TYPE=0
OPTIMAL_AT=-1
EXP_BASE=2.0

def loss_weights(N):
    if FUNC_TYPE == 0: # exponential spacing
        return anytime_loss.at_func(N, func=lambda x:2**x)
    elif FUNC_TYPE == 1: # square spacing
        return anytime_loss.at_func(N, func=lambda x:x**2)
    elif FUNC_TYPE == 2: #optimal at ?
        return anytime_loss.optimal_at(N, OPTIMAL_AT)
    elif FUNC_TYPE == 3: #exponential weights
        return anytime_loss.exponential_weights(N, base=EXP_BASE)
    else:
        raise NameError('func type must be either 0: exponential or 1: square\
            or 2: optimal at --opt_at, or 3: exponential weight with base --base')

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    if NUM_CLASSES == 10:
        ds = dataset.Cifar10(train_or_test)
    elif NUM_CLASSES == 100:
        ds = dataset.Cifar100(train_or_test)
    else:
        raise ValueError('Number of classes must be set to 10(default) or 100 for CIFAR')
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    vcs = []
    total_units = NUM_RES_BLOCKS * NUM_UNITS * WIDTH
    weights = loss_weights(total_units)
    unit_idx = 0
    for bi in range(NUM_RES_BLOCKS):
        for ui in range(NUM_UNITS):
            for wi in range(WIDTH):
                weight = weights[unit_idx]
                unit_idx += 1
                if weight > 0:
                    scope_name = 'res{}.{:02d}.{}.eval/'.format(bi, ui, wi)
                    vcs.append(ClassificationError(\
                        wrong_tensor_name=scope_name+'incorrect_vector:0', 
                        summary_name=scope_name+'val_err'))

    logger.info('weights: {}'.format(weights))
    lr = get_scalar_var('learning_rate', 0.01, summary=True)
    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (250, 0.0002)])
        ],
        model=Model(NUM_UNITS,WIDTH,INIT_CHANNEL,NUM_CLASSES,weights),
        steps_per_epoch=steps_per_epoch,
        max_epoch=300,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=BATCH_SIZE)
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=NUM_UNITS)
    parser.add_argument('-w', '--width',
                        help='width of the network',
                        type=int, default=WIDTH)
    parser.add_argument('-c', '--init_channel',
                        help='channel at beginning of each width of the network',
                        type=int, default=INIT_CHANNEL)
    parser.add_argument('--num_classes', help='Number of classes', 
                        type=int, default=NUM_CLASSES)
    parser.add_argument('-b', '--base', 
                        help='Exponential base',
                        type=np.float32, default=EXP_BASE)
    parser.add_argument('--opt_at', help='Optimal at', type=int, default=OPTIMAL_AT)
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=0)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    NUM_UNITS = args.num_units
    WIDTH = args.width
    INIT_CHANNEL = args.init_channel
    FUNC_TYPE = args.func_type
    NUM_CLASSES = args.num_classes
    OPTIMAL_AT = args.opt_at
    EXP_BASE = args.base

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if os.getenv('LOG_DIR') is None:
        logger.auto_set_dir()
    else:
        logger.auto_set_dir(log_root = os.environ['LOG_DIR'])
    if os.getenv('DATA_DIR') is not None:
        os.environ['TENSORPACK_DATASET'] = os.environ['DATA_DIR']

    logger.info("Parameters: n= {}, w= {}, c= {}, batch_size={}, -f= {}, -b= {}, --opt_at= {}".format(NUM_UNITS,\
        WIDTH, INIT_CHANNEL, BATCH_SIZE, FUNC_TYPE, EXP_BASE, OPTIMAL_AT))

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
