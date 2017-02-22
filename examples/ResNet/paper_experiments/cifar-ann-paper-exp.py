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
from tensorpack.utils import utils

from tensorflow.contrib.layers import variance_scaling_initializer

"""
"""
DO_VALID=False

BATCH_SIZE = 128
NUM_RES_BLOCKS = 3
NUM_UNITS = 5
WIDTH = 1
INIT_CHANNEL = 16
NUM_CLASSES = 10

FUNC_TYPE=2
OPTIMAL_AT=-1
EXP_BASE=2.0

STOP_GRADIENTS=False
STOP_GRADIENTS_PARTIAL=False

class Model(ModelDesc):

    def __init__(self, n, width, init_channel, num_classes, weights):
        super(Model, self).__init__()
        self.n = n
        self.width = width
        self.init_channel = init_channel
        self.num_classes = num_classes
        self.weights = weights

    def _get_inputs(self):
        return [InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                InputVar(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0 - 1

        def conv(name, l, channel, stride):
            kernel = 3
            stddev = np.sqrt(2.0/kernel/kernel/channel)
            return Conv2D(name, l, channel, kernel, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=stddev))

        def residual(name, l_feats, increase_dim=False):
            shape = l_feats[0].get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            l_mid_feats = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.mid') as scope:
                    l = BatchNorm('bn0', l_feats[w])
                    # The first round doesn't use relu per pyramidial deep net
                    # l = tf.nn.relu(l)
                    if w == 0:
                        merged_feats = l
                    else:
                        merged_feats = tf.concat(3, [merged_feats, l], name='concat_mf')
                    l = conv('conv1', merged_feats, out_channel, stride1)
                    l = BatchNorm('bn1', l)
                    l = tf.nn.relu(l)
                    l_mid_feats.append(l)

            l_end_feats = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.end') as scope:
                    l = l_mid_feats[w]
                    if w == 0:
                        merged_feats = l
                    else: 
                        merged_feats = tf.concat(3, [merged_feats, l], name='concat_ef')
                    ef = conv('conv2', merged_feats, out_channel, 1)
                    # The second conv need to be BN before addition.
                    ef = BatchNorm('bn2', ef)
                    l = l_feats[w]
                    if increase_dim:
                        l = AvgPooling('pool', l, 2)
                        l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])
                    ef += l
                    l_end_feats.append(ef)
            return l_end_feats

        def row_sum_predict(name, l_feats, out_dim):
            l_logits = []
            var_list = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.predict') as scope:
                    l = tf.nn.relu(l_feats[w])
                    l = GlobalAvgPooling('gap', l)
                    if w == 0:
                        merged_feats = l
                    else:
                        merged_feats = tf.concat(1, [merged_feats, l], name='concat')
                    logits, vl = FullyConnected('linear', merged_feats, out_dim, \
                                                nl=tf.identity, return_vars=True)
                    var_list.extend(vl)
                    #if w != 0:
                    #    logits += l_logits[-1]
                    l_logits.append(logits)
            return l_logits, var_list

        def cost_and_eval(name, l_logits, label):
            l_costs = []
            l_wrong = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.eval') as scope:
                    logits = l_logits[w]
                    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
                    cost = tf.reduce_mean(cost, name='cross_entropy_loss')
                    add_moving_summary(cost)

                    wrong = prediction_incorrect(logits, label)
                    wrong = tf.reduce_mean(wrong, name='train_error')
                    add_moving_summary(wrong)

                    l_costs.append(cost)
                    l_wrong.append(wrong)
            return l_costs, l_wrong

        l_feats = [] 
        for w in range(self.width):
            with tf.variable_scope('init_conv'+str(w)) as scope:
                l = conv('conv0', image, self.init_channel, 1) 
                #l = BatchNorm('bn0', l)
                #l = tf.nn.relu(l)
                l_feats.append(l)

        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = 0
        cost = 0
        unit_idx = 0
        for res_block_i in range(NUM_RES_BLOCKS):
            for k in range(self.n):
                scope_name = 'res{}.{:02d}'.format(res_block_i, k)
                l_feats = \
                    residual(scope_name, l_feats, 
                             increase_dim=(k==0 and res_block_i > 0))
                l_logits, var_list = row_sum_predict(scope_name, l_feats, self.num_classes) 
                l_costs, l_wrong = cost_and_eval(scope_name, l_logits, label)

                is_last_row = res_block_i == NUM_RES_BLOCKS-1 and k==self.n-1
                for ci, c in enumerate(l_costs):
                    cost_weight = self.weights[unit_idx]
                    unit_idx += 1
                    if cost_weight > 0:
                        cost += cost_weight * c
                        # Regularize weights from FC layers. Should use 
                        # regularize_cost to get the weights using variable names
                        wd_cost += cost_weight * wd_w * tf.nn.l2_loss(var_list[2*ci])
                        if STOP_GRADIENTS_PARTIAL and not is_last_row: 
                            l = l_feats[ci]
                            l = (1 - SG_GAMMA) * tf.stop_gradient(l) + SG_GAMMA * l
                            l_feats[ci] = l


        # weight decay on all W on conv layers
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), \
                         name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')


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
    elif FUNC_TYPE == 5: # sieve
        return anytime_loss.sieve_loss_weights(N)
    elif FUNC_TYPE == 6: # linear
        return anytime_loss.linear(N, a=0.25, b=1.0)
    elif FUNC_TYPE == 7: # half constant, half optimal at -1
        return anytime_loss.half_constant_half_optimal(N, -1)
    elif FUNC_TYPE == 8: # quater constant, half optimal
        return anytime_loss.quater_constant_half_optimal(N)
    else:
        raise NameError('func type must be either 0: exponential or 1: square\
            or 2: optimal at --opt_at, or 3: exponential weight with base --base')

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    if NUM_CLASSES == 10:
        ds = dataset.Cifar10(train_or_test, do_validation=DO_VALID)
    elif NUM_CLASSES == 100:
        ds = dataset.Cifar100(train_or_test, do_validation=DO_VALID)
    else:
        raise ValueError('Number of classes must be set to 10(default) or 100 for CIFAR')
    logger.info("Data {} has {} samples".format(train_or_test, len(ds.data)))
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
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
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
    parser.add_argument('--base', 
                        help='Exponential base',
                        type=np.float32, default=EXP_BASE)
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=OPTIMAL_AT)
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=0)
    parser.add_argument('--do_validation', help='Whether to do validation',
                        type=bool, default=DO_VALID)
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
    DO_VALID = args.do_validation

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print args.log_dir
    logger.auto_set_dir(log_root=args.log_dir)
    utils.set_dataset_path(path=args.data_dir, auto_download=False)

    logger.info("Parameters: n= {}, w= {}, c= {}, batch_size={}, -f= {}, -b= {}, --opt_at= {}".format(NUM_UNITS,\
        WIDTH, INIT_CHANNEL, BATCH_SIZE, FUNC_TYPE, EXP_BASE, OPTIMAL_AT))

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
