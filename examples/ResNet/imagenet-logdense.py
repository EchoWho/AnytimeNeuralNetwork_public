#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os

from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.tfutils.symbolic_functions import *
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import anytime_loss, logger, utils
from tensorpack.callbacks import Exp3CPU

"""
"""
NUM_CLASSES=1000
DO_VALID=False

MODEL_DIR=None

TOTAL_BATCH_SIZE=256
NR_GPU=None
BATCH_SIZE=64

INPUT_SHAPE = 224

DEPTH=None
GROWTH_RATE=32
NUM_UNITS_PER_STACK=9

# For other loss weight assignments
FUNC_TYPE=5
OPTIMAL_AT=-1
EXP_BASE=2.0

SAMLOSS=0  
EXP3_GAMMA=0.3
SUM_RAND_RATIO=2.0
LAST_REWARD_RATE = 0.85

TRACK_GRADIENTS=False

IS_TOY=False

# Layer configs for imagenet
cfg = {
    121: ([6, 12, 24, 16], 58),
    169: ([6, 12, 32, 32], 82),
    201: ([6, 12, 48, 32], 98),
    161: ([6, 12, 36, 24], 78)}

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
    def __init__(self, n, growth_rate):
        super(Model, self).__init__()
        self.n = n
        self.growth_rate = growth_rate
        self.reduction_rate = 1

    def _get_inputs(self):
        return [InputVar(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputVar(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        exponential = 1
        exponentials = []
        while exponential < 2000:
            exponentials.append(exponential)
            exponential *= 2

        def compute_block(pls, n_units):
            i = len(pls) - 1
            for ui in range(n_units):
                i += 1
                with tf.variable_scope('layer_{:03d}'.format(i-1)):
                    selected_feats = [ pls[i - e] \
                        for e in exponentials if i - e >= 0]
                    
                    # debug logging only:
                    selected_feats_idx = [ i - e \
                        for e in exponentials if i - e >= 0]
                    print "unit_idx = {}, len past_feats = {}, selected_feats: {}".format(i,
                        len(pls), selected_feats_idx)
                    
                    l = tf.concat(3, selected_feats, name='concat')
                    l = (LinearWrap(l)
                        .BNReLU('bnrelu')
                        .Conv2D('conv1x1', 4 * self.growth_rate, 1, stride=1, nl=BNReLU)
                        .Conv2D('conv3x3', self.growth_rate, 3, stride=1)())

                    pls.append(l)
            return pls

        def compute_transition(pls, trans_idx):
            new_pls = []
            for pli, pl in enumerate(pls):
                with tf.variable_scope('transit_{:02d}_{:02d}'.format(trans_idx, pli)): 
                    new_pls.append((LinearWrap(pl)
                        .BNReLU('bnrelu')
                        .Conv2D('conv', self.growth_rate, 1, stride=1)
                        .AvgPooling('pool', 2, padding='SAME')()))
            return new_pls

        def compute_logits(l, name):
            with tf.variable_scope(name):
                logits = (LinearWrap(l)
                    .BNReLU('bnrelu')
                    .GlobalAvgPooling('gap')
                    .FullyConnected('linear', NUM_CLASSES, nl=tf.identity)())
            return logits

        wd_cost = 0
        total_cost = 0

        block_n_units = cfg[self.n][0]
        n_predictions = cfg[self.n][1]

        with argscope(Conv2D, nl=tf.identity, use_bias=False, 
                      W_init=variance_scaling_initializer(mode='FAN_OUT')):
            l_preprocess = (LinearWrap(image)
                .Conv2D('conv0', 2 * self.growth_rate, 7, stride=2, nl=BNReLU)
                .MaxPooling('pool0', 3, stride=2, padding='SAME')())
            pls = [l_preprocess]
            pls = compute_block(pls, block_n_units[0])
            pls = compute_transition(pls, 0) 
            pls = compute_block(pls, block_n_units[1])
            pls = compute_transition(pls, 1) 
            pls = compute_block(pls, block_n_units[2])
            pls = compute_transition(pls, 2) 
            pls = compute_block(pls, block_n_units[3])

        pls = pls[1:]
        assert n_predictions == len(pls), n_predictions

        weights = loss_weights(n_predictions)
        logger.info("sampling loss with method {}".format(SAMLOSS))
        if SAMLOSS > 0:
            ls_K = np.sum(np.asarray(weights) > 0)
            select_idx = tf.get_variable("select_idx", (), tf.int32,
                initializer=tf.constant_initializer(ls_K - 1), trainable=False)
            for i in range(ls_K):
                weight_i = tf.cast(tf.equal(select_idx, i), tf.float32, name='weight_{}'.format(i))
                add_moving_summary(weight_i)

        loss = 0.0
        anytime_idx = -1
        online_learn_rewards = []
        last_reward = None
        max_reward = 0.0

        for i, l in enumerate(pls):
            cost_weight = weights[i]
            if cost_weight > 0:
                anytime_idx += 1
                with tf.variable_scope('pred_{:03d}'.format(i)) as scope:
                    logits = compute_logits(l, 'logits')
                    lossi = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
                    lossi = tf.reduce_mean(lossi, name='xentropy-loss')

                    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
                    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

                    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
                    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
                
                add_weight = 0
                if SAMLOSS > 0:
                    add_weight = tf.cond(tf.equal(anytime_idx, select_idx),
                        lambda: tf.constant(weights[-1] * 2.0, dtype=tf.float32),
                        lambda: tf.constant(0, dtype=tf.float32))
                if SUM_RAND_RATIO > 0:
                    loss += (cost_weight + add_weight / SUM_RAND_RATIO) * lossi
                else:
                    loss += add_weight * lossi
                
                if not last_reward is None:
                    reward = 1.0 - lossi / last_reward
                    max_reward = tf.maximum(reward, max_reward)
                    online_learn_rewards.append(tf.multiply(reward, 1.0, 
                        name='reward_{:03d}'.format(anytime_idx-1)))
                if i == n_predictions - 1:
                    reward = max_reward * LAST_REWARD_RATE
                    online_learn_rewards.append(tf.multiply(reward, 1.0,
                        name='reward_{:03d}'.format(anytime_idx)))
                last_reward = lossi 
        #end for each layer/prediction

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
    return ds

def get_config():
    # prepare dataset
    if IS_TOY:
        dataset_train = get_data('toy_train')
        dataset_val = get_data('toy_validation')
    else:
        dataset_train = get_data('train')
        dataset_val = get_data('validation')
    steps_per_epoch = dataset_train.size() // NR_GPU


    vcs = []
    n_predictions = cfg[DEPTH][1]
    weights = loss_weights(n_predictions)
    logger.info("weights: {}".format(weights))
    unit_idx = 0
    for i in range(n_predictions):
        if weights[i] > 0:
            scope_name = 'pred_{:03d}/'.format(i)
            vcs.extend([
                ClassificationError(scope_name+'wrong-top1', 
                                    scope_name+'val-error-top1'),
                ClassificationError(scope_name+'wrong-top5', 
                                    scope_name+'val-error-top5')])

    if SAMLOSS > 0:
        ls_K = np.sum(np.asarray(weights) > 0)
        reward_names = [ 'tower0/reward_{:03d}:0'.format(i) for i in range(ls_K)]
        exp3_callback = Exp3CPU(ls_K, EXP3_GAMMA, 
                                'select_idx:0', reward_names)
        exp3_callbacks = [ exp3_callback ]
    else:
        exp3_callbacks = []

    lr = tf.Variable(0.01, trainable=False, name='learning_rate')
    tf.summary.scalar('learning_rate', lr)
    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        callbacks=[
            StatPrinter(),
            ModelSaver(checkpoint_dir=MODEL_DIR),
            InferenceRunner(dataset_val,
                [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (30, 0.01), (60, 1e-3), (85, 1e-4), (95, 1e-5)])
        ] + exp3_callbacks,
        model=Model(n=DEPTH, growth_rate=GROWTH_RATE),
        steps_per_epoch=steps_per_epoch,
        max_epoch=110,
    )

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES=""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='ILSVRC dataset dir')
    parser.add_argument('--log_dir', help='log_dir for stdout')
    parser.add_argument('--model_dir', help='dir for saving models')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=-1)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=TOTAL_BATCH_SIZE)
    parser.add_argument('-f', '--func_type', help='type of loss weight', 
                        type=int, default=FUNC_TYPE)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, \
                              i.e., number of units per prediction',
                        type=int, default=NUM_UNITS_PER_STACK)
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=121, choices=[121, 169, 201, 161])
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=OPTIMAL_AT)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--samloss', help='Method to Sample losses to update',
                        type=int, default=SAMLOSS)
    parser.add_argument('--exp_gamma', help='Gamma for exp3 in sample loss',
                        type=np.float32, default=EXP3_GAMMA)
    parser.add_argument('--sum_rand_ratio', help='frac{Sum weight}{randomly selected weight}',
                        type=np.float32, default=SUM_RAND_RATIO)
    parser.add_argument('--track_grads', help='Whether to track gradient l2 of each loss',
                        type=bool, default=TRACK_GRADIENTS)
    parser.add_argument('--growth_rate', help='growth rate k for log dense',
                        type=int, default=GROWTH_RATE)
    parser.add_argument('--is_toy', help='Whether to have data size of 1024',
                        type=bool, default=IS_TOY)
    parser.add_argument('--last_reward_rate', help='rate of last reward in comparison to the max',
                        type=np.float32, default=LAST_REWARD_RATE)

    args = parser.parse_args()
    MODEL_DIR = args.model_dir
    logger.auto_set_dir(log_root=args.log_dir)
    utils.set_dataset_path(path=args.data_dir, auto_download=False)

    TOTAL_BATCH_SIZE = args.batch_size
    FUNC_TYPE = args.func_type
    NUM_UNITS_PER_STACK = args.stack
    DEPTH = args.depth
    OPTIMAL_AT = args.opt_at

    GROWTH_RATE = args.growth_rate

    SAMLOSS = args.samloss
    EXP3_GAMMA = args.exp_gamma
    SUM_RAND_RATIO = args.sum_rand_ratio
    LAST_REWARD_RATE = args.last_reward_rate
    
    TRACK_GRADIENTS = args.track_grads
    IS_TOY = args.is_toy


    NR_GPU = 1
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        gpus = os.environ['CUDA_VISIBLE_DEVICES']
        NR_GPU = len(gpus.split(','))
    if args.nr_gpu > 0:
        NR_GPU = args.nr_gpu
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
