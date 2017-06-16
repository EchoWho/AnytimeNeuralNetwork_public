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

ADDITIONAL_CONV=False

# DISTILL variable
TEMPERATURE=8
HARD_ONLY=False

# ANN period
NUM_UNITS_PER_STACK=2

# TODO move this to utils
# Control loss weight assignment 
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

        mask_ann_indices = np.asarray(cost_weights) > 0
        ls_K = np.sum(mask_ann_indices)
        unit_idx_to_ann_idx = np.cumsum(mask_ann_indices) - 1
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

        def feat_map_to_1x1_feat(l):
            ch_in = l.get_shape().as_list()[1] 
            if ADDITIONAL_CONV:
                l = Conv2D('conv1x1', l, ch_in, kernel_shape=1)
                l = BatchNorm('bn_f2p0', l)
                l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap_1x1', l)
            return l
               
        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling, MaxPooling], 
                      data_format='NCHW'),\
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3, 
                         W_init=variance_scaling_initializer(mode='FAN_OUT')):
            unit_idx = -1
            # regularzation weight
            wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                              480000, 0.2, True)

            boundaries = [50000 * 100, 50000 * 150, 50000 * 200]
            boundaries = [np.int64(boundary) for boundary in boundaries]
            hard_rate = tf.train.piecewise_constant(get_global_step_var(), \
                boundaries=boundaries, \
                values=[1.0, 0.84, 0.67, 0.5])
            for bi in list(range(NUM_RES_BLOCKS)):
                ##### Transition / initial layer 
                if bi == 0:
                    with tf.variable_scope('init_conv') as scope:
                        l = Conv2D('conv', image, self.init_channel) 
                        l_merged_feats = []
                        l_feats = [l]
                        ll_feats = [l_feats]
                else:
                    new_l_feats = []
                    for pfi, pf in enumerate(l_feats):
                        with tf.variable_scope('trans_{}_{}'.format(bi,pfi)) as scope:
                            l = BatchNorm('bn'.format(bi, pfi), pf)
                            l = tf.nn.relu(l)
                            ch_in = l.get_shape().as_list()[1]
                            l = Conv2D('conv'.format(bi, pfi), l, \
                                       ch_in // self.reduction_rate) 
                            l = tf.nn.relu(l)
                            l = AvgPooling('pool', l, 2)
                            new_l_feats.append(l)
                    l_feats = new_l_feats
                    ll_feats.append(l_feats)
                
                ###### Dense block features
                for k in list(range(self.n)):
                    unit_idx += 1
                    scope_name = 'dense{}.{:02d}'.format(bi, k)
                    is_last_node = k == self.n - 1 and bi == NUM_RES_BLOCKS - 1
                    with tf.variable_scope(scope_name) as scope:
                        if is_last_node:
                            selected_indices = list(range(unit_idx+1))
                        else:
                            selected_indices = log_select_indices(unit_idx)
                        logger.info("unit_idx = {}, len l_feats = {}, selected_feats: {}".format(unit_idx, len(l_feats), selected_indices))
                        selected_feats = [l_feats[s_idx] for s_idx in selected_indices]
                        l = tf.concat(selected_feats, 1, name='concat')
                        #l = BatchNorm('bn0', l)
                        #l = tf.nn.relu(l)
                        #l = Conv2D('conv0', l, 
                        #            self.growth_rate * self.bottleneck_width, kernel_shape=1)
                        l = BatchNorm('bn1', l)
                        merged_feats = tf.nn.relu(l)
                        l_merged_feats.append(merged_feats)
                        l = Conv2D('conv1', merged_feats, self.growth_rate)
                        l_feats.append(l)

                        if is_last_node:
                            l = BatchNorm('bn_last', l)
                            l = tf.nn.relu(l)
                            merged_feats = tf.concat([l_merged_feats[-1], l], \
                                                     1, name='concat')
                            l_merged_feats.append(merged_feats)
                    #end scope with dense{}{}
                # end for each unit
            #end for each dense block
            
            wd_cost = 0
            total_cost = 0
            prev_logits = None
            # compute the final prediction first:
            for unit_idx in list(reversed(range(total_units))):
                cost_weight = cost_weights[unit_idx]        
                is_last_node = unit_idx == total_units - 1
                if cost_weight > 0:
                    ann_idx = np.int32(unit_idx_to_ann_idx[unit_idx])
                    print "ann_idx = {}, unit_idx = {}".format(ann_idx, unit_idx)
                    scope_name = 'pred_{:03d}'.format(unit_idx)
                    with tf.variable_scope(scope_name) as scope:
                        merged_feats = l_merged_feats[unit_idx+1]
                        l = feat_map_to_1x1_feat(merged_feats)
                        logits = FullyConnected('linear', l, \
                            out_dim=NUM_CLASSES, nl=tf.identity)

                        # Distills:
                        if is_last_node:
                            target_logits = logits / TEMPERATURE
                            target_prob = \
                                tf.stop_gradient(tf.nn.softmax(target_logits, \
                                    name='final_prob'))
                            #target_prob = tf.Print(target_prob, [target_logits, target_prob])
                            cost = \
                                tf.nn.sparse_softmax_cross_entropy_with_logits(\
                                    logits=target_logits, labels=label)
                        else:
                            cost_hard = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                                logits=logits, labels=label)
                            if HARD_ONLY:
                                cost = cost_hard
                            else:
                                cost_soft = \
                                    tf.nn.softmax_cross_entropy_with_logits(\
                                        logits=logits / TEMPERATURE, labels=target_prob)\
                                    #* TEMPERATURE**2
                                cost =  cost_soft * (1-hard_rate) + cost_hard * hard_rate

                        #cost = tf.Print(cost, [target_logits, target_prob])

                        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

                        wrong = prediction_incorrect(logits, label)
                        tra_err = tf.reduce_mean(wrong, name='train_error')

                        # because cost_weight >0, ann_idx inc
                        add_weight = 0
                        if ls_K > 1: # use SAMLOSS == 6 to sample weights
                            add_weight = tf.cond(tf.equal(ann_idx, select_idx),
                                lambda: tf.constant(cost_weights[-1], dtype=tf.float32),
                                lambda: tf.constant(0, dtype=tf.float32))
                        total_cost += (cost_weight + add_weight) * cost

                        # cost_weight adjusted regularzation:
                        wd_cost += cost_weight * wd_w * tf.nn.l2_loss(logits.variables.W) 

                        add_moving_summary(cost)
                        add_moving_summary(tra_err)
                    # end var-scope of a unit
                # end if cost_weight > 0 
            # end for unit 
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
            scope_name = 'pred_{:03d}/'.format(unit_idx)
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
    parser.add_argument('--additional_conv', 
        help='whether to use 1x1 conv b/f fully connected prediction', 
        type=bool, default=ADDITIONAL_CONV)
    parser.add_argument('--temperature', help='temperature for logits', 
                        type=float, default=TEMPERATURE)
    parser.add_argument('--hard_only', help='Whether to use hard target only',
                        type=bool, default=HARD_ONLY)
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
    ADDITIONAL_CONV = args.additional_conv
    TEMPERATURE = args.temperature
    HARD_ONLY = args.hard_only
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
