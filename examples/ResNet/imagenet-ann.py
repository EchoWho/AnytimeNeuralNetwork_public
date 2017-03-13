#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
Training code of Pre-Activation version of ResNet on ImageNet.
It mainly follows the setup in fb.resnet.torch, and get similar performance.
"""

TOTAL_BATCH_SIZE = 256
BATCH_SIZE=None
NR_GPU=None
INPUT_SHAPE = 224
DEPTH = None

MODEL_DIR=None

# For other loss weight assignments
FUNC_TYPE=5
OPTIMAL_AT=-1
EXP_BASE=2.0

# anytime loss skip (num units per stack/prediction)
NUM_UNITS_PER_STACK=2

def loss_weights(N):
    if FUNC_TYPE == 0: # exponential spacing
        return anytime_loss.at_func(N, func=lambda x:2**x)
    elif FUNC_TYPE == 1: # square spacing
        return anytime_loss.at_func(N, func=lambda x:x**2)
    elif FUNC_TYPE == 2: #optimal at ?
        return anytime_loss.optimal_at(N, OPTIMAL_AT)
    elif FUNC_TYPE == 3: #exponential weights
        return anytime_loss.exponential_weights(N, base=EXP_BASE)
    elif FUNC_TYPE == 4: #constant weights
        return anytime_loss.constant_weights(N) 
    elif FUNC_TYPE == 5: # sieve with stack
        return anytime_loss.stack_loss_weights(N, NUM_UNITS_PER_STACK)
    elif FUNC_TYPE == 6: # linear
        return anytime_loss.linear(N, a=0.25, b=1.0)
    elif FUNC_TYPE == 7: # half constant, half optimal at -1
        return anytime_loss.half_constant_half_optimal(N, -1)
    elif FUNC_TYPE == 8: # quater constant, half optimal
        return anytime_loss.quater_constant_half_optimal(N)
    else:
        raise NameError('func type must be either 0: exponential or 1: square\
            or 2: optimal at --opt_at, or 3: exponential weight with base --base')

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputVar(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputVar(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[-1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[-1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layers(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                ls = []
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                    ls.append(l)
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                        ls.append(l)
                return ls

        def compute_logits(l, layername):
            with tf.variable_scope(layername):
                logits = (LinearWrap(l)
                      .BNReLU('bnlast')
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, nl=tf.identity)())
            return logits

        cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = cfg[DEPTH]
        cfg_N = {18:8, 34:16, 50:16, 101:33}
        N = cfg_N[DEPTH]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')):
            
            l_preprocess = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')())
            ls_grp0 = layers(l_preprocess, 'group0', block_func, 64, defs[0], 1, first=True)
            ls_grp1 = layers(ls_grp0[-1], 'group1', block_func, 128, defs[1], 2)
            ls_grp2 = layers(ls_grp1[-1], 'group2', block_func, 256, defs[2], 2)
            ls_grp3 = layers(ls_grp2[-1], 'group3', block_func, 512, defs[3], 2)

        ls = ls_grp0 + ls_grp1 + ls_grp2 + ls_grp3 
        assert N == len(ls), N
        weights = loss_weights(N)
        loss = 0
        for i in range(N):
            if weights[i] > 0:
                with tf.variable_scope('pred_{:02d}'.format(i)) as scope:
                    logits = compute_logits(ls[i], 'logits')
                    lossi = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
                    lossi = tf.reduce_mean(lossi, name='xentropy-loss')

                    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
                    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

                    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
                    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

                    loss += weights[i] * lossi

        wd_cost = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        loss = tf.identity(loss, name='sum_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.ILSVRC12TFRecord(args.data_dir, 
                                  train_or_test, 
                                  BATCH_SIZE, 
                                  height=INPUT_SHAPE, 
                                  width=INPUT_SHAPE)
    #image_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    #image_std = np.array([0.229, 0.224, 0.225], dtype='float32')
    #imgaug.Saturation(0.4)
    #imgaug.Lighting(0.1,
    #             eigval=[0.2175, 0.0188, 0.0045],
    #             eigvec=[[-0.5675, 0.7192, 0.4009],
    #                     [-0.5808, -0.0045, -0.8140],
    #                     [-0.5836, -0.6948, 0.4203]]
    #             )
    return ds


def get_config():
    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size() // NR_GPU
    dataset_val = get_data('validation')

    vcs = []
    cfg_N = { 18:8, 34:16, 50:16, 101:33 }
    N = cfg_N[DEPTH]
    weights = loss_weights(N)
    logger.info("weights: {}".format(weights))
    unit_idx = 0
    for i in range(N):
        if weights[i] > 0:
            scope_name = 'pred_{:02d}/'.format(i)
            vcs.extend([
                ClassificationError(scope_name+'wrong-top1', 
                                    scope_name+'val-error-top1'),
                ClassificationError(scope_name+'wrong-top5', 
                                    scope_name+'val-error-top5')])

    lr = get_scalar_var('learning_rate', 0.1, summary=True)
    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        callbacks=[
            ModelSaver(checkpoint_dir=MODEL_DIR),
            InferenceRunner(dataset_val, vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=110,
    )

def eval_on_ILSVRC12(model_file, data_dir):
    ds = get_data('val')
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_file),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir', help='ILSVRC dataset dir')
    parser.add_argument('--log_dir', help='log_dir for stdout')
    parser.add_argument('--model_dir', help='dir for saving models')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size', help='batch_size', type=int, default=TOTAL_BATCH_SIZE)
    parser.add_argument('-f', '--func_type', help='type of loss weight', 
                        type=int, default=FUNC_TYPE)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, \
                              i.e., number of units per prediction',
                        type=int, default=NUM_UNITS_PER_STACK)
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101])
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=OPTIMAL_AT)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    # directory setup
    MODEL_DIR = args.model_dir
    logger.auto_set_dir(log_root=args.log_dir)

    TOTAL_BATCH_SIZE = args.batch_size
    FUNC_TYPE = args.func_type
    NUM_UNITS_PER_STACK = args.stack
    DEPTH = args.depth
    OPTIMAL_AT = args.opt_at
    
    #if args.eval:
    #    BATCH_SIZE = 128    # something that can run on one gpu
    #    eval_on_ILSVRC12(args.load, args.data_dir)
    #    sys.exit()

    NR_GPU = 1
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        gpus = os.environ['CUDA_VISIBLE_DEVICES']
        NR_GPU = len(gpus.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
