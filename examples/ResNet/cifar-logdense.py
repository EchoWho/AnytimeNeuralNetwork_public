#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import anytime_loss, logger, utils, fs
from tensorpack.callbacks import FixedDistributionCPU
from tensorflow.contrib.layers import variance_scaling_initializer

"""
Modified for generating anytime prediction after
a network is trained. 
added FUNC_TYPE 109 for this.
ModelSaver keep_freq=1 for this
"""
# dataset selection
NUM_CLASSES=10
DO_VALID=False

# fs path set up
MODEL_DIR=None

# model complexity param (dense net param)
BATCH_SIZE=64
NUM_RES_BLOCKS=3
NUM_UNITS=12
GROWTH_RATE=16
INIT_CHANNEL=16

# LOG DENSE specific
LOG_SELECT_METHOD=0
LOG_ANN_SELECT_METHOD=0

# ANN period
NUM_UNITS_PER_STACK=2

# TODO move this to utils
# Control loss weight assignment 
# Also hack: TODO remove hack 109 type is for partially stop grad
FUNC_TYPE=5
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
    elif FUNC_TYPE == 109: # train anytime predictor after the final is fixed through no-backs
        return anytime_loss.stack_loss_weights(N, 
            NUM_UNITS_PER_STACK, 
            anytime_loss.constant_weights)
    else:
        raise NameError('func type must be either 0: exponential or 1: square\
            or 2: optimal at --opt_at, or 3: exponential weight with base --base')

class Model(ModelDesc):
    def __init__(self, n, growth_rate, init_channel):
        super(Model, self).__init__()
        self.n = n
        self.growth_rate = growth_rate
        self.init_channel = init_channel
        #self.bottleneck_width = 4
        self.reduction_rate = 1

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0
        image = tf.transpose(image, [0, 3, 1, 2]) 

        # total depth (block * n) + initial 
        total_units = NUM_RES_BLOCKS * self.n
        cost_weights = loss_weights(total_units) 

        ls_K = np.sum(np.asarray(cost_weights) > 0)
        if ls_K > 1:
            select_idx = tf.get_variable('select_idx', (), tf.int32,
                initializer=tf.constant_initializer(ls_K-1), trainable=False)
            for i in range(ls_K):
                weight_i = tf.cast(tf.equal(select_idx, i), \
                    tf.float32, name='weight_{}'.format(i))
                add_moving_summary(weight_i)
        
        exponential = 1
        exponentials = []
        while exponential < total_units + 1:
            exponentials.append(exponential)
            exponential *= 2

        def log_select_indices(u_idx):
            if LOG_SELECT_METHOD == 0:
                diffs = exponentials 
            elif LOG_SELECT_METHOD == 1:
                diffs = list(range(1, int(np.log2(u_idx + 1)) + 1))
            elif LOG_SELECT_METHOD == 2:
                diffs = list(range(1, int(np.log2(total_units + 1)) + 1))
            elif LOG_SELECT_METHOD == 3:
                delta = int(np.log2(total_units + 1))
                diffs = list(range(1, total_units + 1, delta)) 
            indices = [u_idx - e + 1 \
                for e in diffs if u_idx - e + 1 >= 0 ]
            return indices
               
        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling, MaxPooling], 
                      data_format='NCHW'),\
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3, 
                         W_init=variance_scaling_initializer(mode='FAN_OUT')):
            past_feats = []
            prev_logits = None
            unit_idx = -1
            anytime_idx = -1
            wd_cost = 0
            total_cost = 0
            # regularzation weight
            wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                              480000, 0.2, True)
            for bi in range(NUM_RES_BLOCKS):
                ##### Transition / initial layer 
                if bi == 0:
                    with tf.variable_scope('init_conv') as scope:
                        l = Conv2D('conv', image, self.init_channel) 
                    past_feats.append(l)
                else:
                    new_past_feats = []
                    with tf.variable_scope('trans_{}'.format(bi)) as scope:
                        for pfi, pf in enumerate(past_feats):
                            l = BatchNorm('bn_{}_{}'.format(bi, pfi), pf)
                            l = tf.nn.relu(l)
                            ch_in = l.get_shape().as_list()[1]
                            l = Conv2D('conv_{}_{}'.format(bi, pfi), l, \
                                       ch_in // self.reduction_rate) 
                            l = tf.nn.relu(l)
                            l = AvgPooling('pool', l, 2)
                            new_past_feats.append(l)
                    past_feats = new_past_feats
                
                ###### Dense block
                for k in range(self.n):
                    unit_idx += 1
                    cost_weight = cost_weights[unit_idx]
                    scope_name = 'dense{}.{:02d}'.format(bi, k)
                    is_last_node = k == self.n - 1 and bi == NUM_RES_BLOCKS - 1
                    with tf.variable_scope(scope_name) as scope:
                        if is_last_node:
                            selected_indices = list(range(unit_idx+1))
                        else:
                            selected_indices = log_select_indices(unit_idx)
                        logger.info("unit_idx = {}, len past_feats = {}, selected_feats: {}".format(unit_idx, len(past_feats), selected_indices))
                        selected_feats = [past_feats[s_idx] for s_idx in selected_indices]
                        l = tf.concat(selected_feats, 1, name='concat')
                        #l = BatchNorm('bn0', l)
                        #l = tf.nn.relu(l)
                        #l = Conv2D('conv0', l, 
                        #            self.growth_rate * self.bottleneck_width, kernel_shape=1)
                        l = BatchNorm('bn1', l)
                        merged_feats = tf.nn.relu(l)
                        l = Conv2D('conv1', merged_feats, self.growth_rate)
                        past_feats.append(l)
                        
                        if cost_weight > 0:
                            #print "Stop gradient at {}".format(scope_name)
                            #merged_feats = tf.stop_gradient(merged_feats)
                            if not is_last_node and FUNC_TYPE == 109:
                                l = tf.stop_gradient(l)
                            l = BatchNorm('bn_pred', l)
                            l = tf.nn.relu(l)
                            l = GlobalAvgPooling('gap', l)
                            if LOG_ANN_SELECT_METHOD == 0:
                                l_prev = GlobalAvgPooling('gap_prev', merged_feats)
                                l = tf.concat([l, l_prev], 1, name='gap_concat')
                            elif LOG_ANN_SELECT_METHOD == 1:
                                l = l
                            logits = FullyConnected('linear', l, out_dim=NUM_CLASSES, nl=tf.identity)

                            if not is_last_node and FUNC_TYPE == 109:
                                if prev_logits is not None:
                                    logits += prev_logits
                                prev_logits = tf.stop_gradient(logits)
                            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
                            cost = tf.reduce_mean(cost, name='cross_entropy_loss')

                            wrong = prediction_incorrect(logits, label)
                            tra_err = tf.reduce_mean(wrong, name='train_error')

                            # because cost_weight >0, anytime_idx inc
                            anytime_idx += 1
                            add_weight = 0
                            if ls_K > 1: # use SAMLOSS == 6 to sample weights
                                add_weight = tf.cond(tf.equal(anytime_idx, select_idx),
                                    lambda: tf.constant(cost_weights[-1], dtype=tf.float32),
                                    lambda: tf.constant(0, dtype=tf.float32))
                            total_cost += (cost_weight + add_weight) * cost

                            # cost_weight adjusted regularzation:
                            wd_cost += cost_weight * wd_w * tf.nn.l2_loss(logits.variables.W) 

                            print '{} {} {}'.format(bi, k, cost_weight)

                            add_moving_summary(cost)
                            add_moving_summary(tra_err)
                    # end var-scope of a unit
                # end for each unit 
            # end for each block
        # END argscope

        # regularize conv
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), \
                         name='wd_cost')
        total_cost = tf.identity(total_cost, name='pred_cost')
        
        add_moving_summary(total_cost, wd_cost)
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([total_cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    if NUM_CLASSES == 10:
        ds = dataset.Cifar10(train_or_test, do_validation=DO_VALID)
    elif NUM_CLASSES == 100:
        ds = dataset.Cifar100(train_or_test, do_validation=DO_VALID)
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
    total_units = NUM_RES_BLOCKS * NUM_UNITS
    weights = loss_weights(total_units)
    unit_idx = -1
    for bi in range(NUM_RES_BLOCKS):
        for ui in range(NUM_UNITS):
            unit_idx += 1
            scope_name = 'dense{}.{:02d}/'.format(bi, ui)
            if weights[unit_idx] > 0:
                vcs.append(ClassificationError(\
                    wrong_tensor_name=scope_name+'incorrect_vector:0', 
                    summary_name=scope_name+'val_err'))

    ls_K = np.sum(np.asarray(weights) > 0)
    if ls_K > 1: 
        ann_cbs = [FixedDistributionCPU(ls_K, 'select_idx:0', weights[weights>0])]
    else:
        ann_cbs = []
    
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(checkpoint_dir=MODEL_DIR),
            InferenceRunner(dataset_test,
                [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (150, 0.01), (225, 0.001)])
        ] + ann_cbs,
        model=Model(n=NUM_UNITS, growth_rate=GROWTH_RATE, init_channel=INIT_CHANNEL),
        steps_per_epoch=steps_per_epoch,
        max_epoch=300,
    )

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES=""
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
    parser.add_argument('--model_dir', help='model_dir position',
                        type=str, default=None)
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=NUM_UNITS)
    parser.add_argument('-g', '--growth_rate',
                        help='number of channel per new layer',
                        type=int, default=GROWTH_RATE)
    parser.add_argument('-c', '--init_channel',
                        help='number of initial channels',
                        type=int, default=INIT_CHANNEL)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, i.e., number of units per prediction',
                        type=int, default=NUM_UNITS_PER_STACK)
    parser.add_argument('--log_method', help='LOG_SELECT_METHOD', 
                        type=int, default=LOG_SELECT_METHOD)
    parser.add_argument('--log_ann_method', help='LOG_ANN_SELECT_METHOD',
                        type=int, default=LOG_ANN_SELECT_METHOD)
    parser.add_argument('--num_classes', help='Number of classes', 
                        type=int, default=NUM_CLASSES)
    parser.add_argument('--do_validation', help='Whether use validation set. Default not',
                        type=bool, default=DO_VALID)
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=FUNC_TYPE)
    parser.add_argument('--base', help='Exponential base',
                        type=np.float32, default=EXP_BASE)
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=OPTIMAL_AT)
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=BATCH_SIZE)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    BATCH_SIZE=args.batch_size
    NUM_UNITS = args.num_units
    GROWTH_RATE = args.growth_rate
    INIT_CHANNEL = args.init_channel
    NUM_UNITS_PER_STACK = args.stack
    LOG_SELECT_METHOD = args.log_method
    LOG_ANN_SELECT_METHOD = args.log_ann_method
    FUNC_TYPE = args.func_type
    EXP_BASE = args.base
    OPTIMAL_AT = args.opt_at
    DO_VALID = args.do_validation
    NUM_CLASSES = args.num_classes

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_log_root(args.log_dir)
    logger.auto_set_dir()
    fs.set_dataset_path(path=args.data_dir, auto_download=False)
    MODEL_DIR = args.model_dir

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
        config.set_tower(tower=map(int, args.gpu.split(',')))
    SyncMultiGPUTrainer(config).train()
