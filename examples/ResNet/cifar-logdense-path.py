#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import anytime_loss, logger, utils

"""
"""
NUM_CLASSES=10
DO_VALID=False

MODEL_DIR=None

BATCH_SIZE=64
NUM_RES_BLOCKS=3
NUM_UNITS=12
GROWTH_RATE=16
INIT_CHANNEL=16

NUM_UNITS_PER_STACK=2

# For other loss weight assignments
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
        return [InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                InputVar(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0 - 1

        def conv(name, l, channel, stride, kernel=3):
            return Conv2D(name, l, channel, kernel, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/kernel/kernel/channel)))
        wd_cost = 0
        total_cost = 0
        total_units = NUM_RES_BLOCKS * self.n
        cost_weights = loss_weights(total_units) 
        unit_idx = -1

        # total depth (block * n) + initial 
        exponential = 1
        exponentials = []
        while exponential < total_units + 1:
            exponentials.append(exponential)
            exponential *= 2
        
        past_feats = []
        for bi in range(NUM_RES_BLOCKS):
            if bi == 0:
                with tf.variable_scope('init_conv') as scope:
                    l = conv('conv', image, self.init_channel, 1) 
                past_feats.append(l)
            else:
                new_past_feats = []
                with tf.variable_scope('trans_{}'.format(bi)) as scope:
                    for pfi, pf in enumerate(past_feats):
                        l = BatchNorm('bn_{}_{}'.format(bi, pfi), pf)
                        l = tf.nn.relu(l)
                        l = conv('conv_{}_{}'.format(bi, pfi), \
                            l, int(l.get_shape().as_list()[3] / self.reduction_rate), 1)
                        l = tf.nn.relu(l)
                        l = AvgPooling('pool', l, 2)
                        new_past_feats.append(l)
                past_feats = new_past_feats

            for k in range(self.n):
                unit_idx += 1
                scope_name = 'dense{}.{}'.format(bi, k)
                with tf.variable_scope(scope_name) as scope:
                    if k < self.n - 1 or bi < NUM_RES_BLOCKS - 1 :
                        selected_feats = [ past_feats[unit_idx - e + 1] \
                            for e in exponentials if unit_idx - e + 1 >= 0]
                        
                        selected_feats_idx = [ unit_idx - e + 1 \
                            for e in exponentials if unit_idx - e + 1 >= 0]
                        print "unit_idx = {}, len past_feats = {}, selected_feats: {}".format(unit_idx,
                            len(past_feats), selected_feats_idx)
                        l = tf.concat(3, selected_feats, name='concat')
                        #l = BatchNorm('bn0', l)
                        #l = tf.nn.relu(l)
                        #l = conv('conv0', l, self.growth_rate * self.bottleneck_width, 1, 1)
                        l = BatchNorm('bn1', l)
                        l = tf.nn.relu(l)
                        l = conv('conv1', l, self.growth_rate, 1)
                        past_feats.append(l)
                    else:
                        merged_feats = None
                        for pfi, pf in enumerate(past_feats):
                            predict_scope_name = 'predict_{}'.format(pfi)
                            with tf.variable_scope(predict_scope_name) as scope:
                                if merged_feats is None:
                                    merged_feats = pf
                                else:
                                    merged_feats = tf.concat(3, [ merged_feats, pf ], name='concat'.format(pfi))

                                cost_weight = cost_weights[pfi] 
                                if cost_weight > 0:
                                    l = BatchNorm('bn_pred', merged_feats)
                                    l = tf.nn.relu(l)
                                    l = GlobalAvgPooling('gap', l)
                                    logits, vl = FullyConnected('linear', l, out_dim=NUM_CLASSES, nl=tf.identity, return_vars=True)
                                    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
                                    cost = tf.reduce_mean(cost, name='cross_entropy_loss')

                                    wrong = prediction_incorrect(logits, label)
                                    tra_err = tf.reduce_mean(wrong, name='train_error')

                                    total_cost += cost_weight * cost

                                    add_moving_summary(cost)
                                    add_moving_summary(tra_err)

        # regularize conv
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.mul(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        total_cost = tf.identity(total_cost, name='pred_cost')
        
        add_moving_summary(total_cost, wd_cost)
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([total_cost, wd_cost], name='cost')


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


    get_global_step_var()
    lr = tf.Variable(0.1, trainable=False, name='learning_rate')
    tf.summary.scalar('learning_rate', lr)

    vcs = []
    total_units = NUM_RES_BLOCKS * NUM_UNITS
    cost_weights = loss_weights(total_units)
    for unit_idx in range(total_units):
        scope_name = 'dense{}.{}/predict_{}/'.format(NUM_RES_BLOCKS - 1, NUM_UNITS - 1, unit_idx)
        if cost_weights[unit_idx] > 0:
            vcs.append(ClassificationError(\
                wrong_tensor_name=scope_name+'incorrect_vector:0', 
                summary_name=scope_name+'val_err'))

    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        callbacks=[
            StatPrinter(),
            ModelSaver(checkpoint_dir=MODEL_DIR),
            InferenceRunner(dataset_test,
                [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (150, 0.01), (225, 0.001)])
        ],
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
                        type=int, default=12)
    parser.add_argument('-g', '--growth_rate',
                        help='number of channel per new layer',
                        type=int, default=GROWTH_RATE)
    parser.add_argument('-c', '--init_channel',
                        help='number of initial channels',
                        type=int, default=INIT_CHANNEL)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, i.e., number of units per prediction',
                        type=int, default=1)
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
    FUNC_TYPE = args.func_type
    EXP_BASE = args.base
    OPTIMAL_AT = args.opt_at
    DO_VALID = args.do_validation
    NUM_CLASSES = args.num_classes

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir(log_root=args.log_dir)
    utils.set_dataset_path(path=args.data_dir, auto_download=False)
    MODEL_DIR = args.model_dir

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
        config.set_tower(tower=map(int, args.gpu.split(',')))
    SyncMultiGPUTrainer(config).train()
