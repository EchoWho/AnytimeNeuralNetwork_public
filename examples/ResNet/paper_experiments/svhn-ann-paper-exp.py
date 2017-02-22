#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-resnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.models import Exp3,HalfEndHalfExp3,RandSelect 
from tensorpack.utils import anytime_loss
from tensorpack.utils import logger
from tensorpack.utils import utils

from tensorflow.contrib.layers import variance_scaling_initializer

"""
"""
# Network structure
BATCH_SIZE = 128
NUM_RES_BLOCKS = 3
NUM_UNITS = 5
WIDTH = 1
INIT_CHANNEL = 16
NUM_CLASSES=10

# anytime loss skip (num units per stack/prediction)
NUM_UNITS_PER_STACK=1
FUNC_TYPE=5
OPTIMAL_AT=-1

# Random loss sample params
SAMLOSS=0
EXP3_GAMMA=0.1
SUM_RAND_RATIO=2.0

# Stop gradients params
STOP_GRADIENTS=False
STOP_GRADIENTS_PARTIAL=False
SG_GAMMA = 0.3

TRACK_GRADIENTS=False

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

        logger.info("sampling loss with method {}".format(SAMLOSS))
        if SAMLOSS > 0:
            ls_K = np.sum(np.asarray(self.weights) > 0)
            if SAMLOSS == 1:
                loss_selector = RandSelect('rand', ls_K)
            elif SAMLOSS == 2:
                loss_selector = Exp3('exp3', ls_K, EXP3_GAMMA)
            elif SAMLOSS == 3:
                loss_selector = HalfEndHalfExp3('hehe3', ls_K, EXP3_GAMMA)
            else:
                raise Exception("Undefined loss selector")
            ls_i, ls_p = loss_selector.sample()
            for i in range(ls_K):
                weight_i = tf.cast(tf.equal(ls_i, i), tf.float32, name='weight_{}'.format(i))
                add_moving_summary(weight_i)

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
        anytime_idx = 0
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
                        anytime_idx += 1
                        add_weight = 0
                        if SAMLOSS > 0:
                            def rand_weight_and_update_ls(loss=c):
                                gs = tf.gradients(loss, tf.trainable_variables()) 
                                reward =  tf.add_n([tf.nn.l2_loss(g) for g in gs if g is not None])
                                loss_selector.update(ls_i, ls_p, reward)
                                return tf.constant(self.weights[-1] * 2, dtype=tf.float32)
                            add_weight = tf.cond(tf.equal(anytime_idx-1, ls_i), 
                                rand_weight_and_update_ls, lambda: tf.zeros(()))
                        if TRACK_GRADIENTS:
                            gs = tf.gradients(c, tf.trainable_variables())
                            reward =  tf.add_n([tf.nn.l2_loss(g) for g in gs if g is not None], 
                                               name='l2_grad_{}'.format(anytime_idx-1))
                            add_moving_summary(reward)
                        cost += (cost_weight + add_weight / SUM_RAND_RATIO) * c
                        # Regularize weights from FC layers. Should use 
                        # regularize_cost to get the weights using variable names
                        wd_cost += cost_weight * wd_w * tf.nn.l2_loss(var_list[2*ci])
                        if STOP_GRADIENTS_PARTIAL and not is_last_row: 
                            l = l_feats[ci]
                            l = (1 - SG_GAMMA) * tf.stop_gradient(l) + SG_GAMMA * l
                            l_feats[ci] = l
                    #endif cost_weight > 0
                #endfor each width
            #endfor each n
        # endfor each block

        # weight decay on all W on conv layers
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), \
                         name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    pp_mean = dataset.SVHNDigit.get_per_pixel_mean()
    if isTrain:
        d1 = dataset.SVHNDigit('train')
        d2 = dataset.SVHNDigit('extra')
        ds = RandomMixData([d1, d2])
    else:
        ds = dataset.SVHNDigit('test')

    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.Brightness(10),
            imgaug.Contrast((0.8, 1.2)),
            imgaug.GaussianDeform(  # this is slow. without it, can only reach 1.9% error
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (40, 40), 0.2, 3),
            imgaug.RandomCrop((32, 32)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
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
    if SAMLOSS > 0:
        lr_schedule = [(1, 0.1), (82, 0.02), (123, 0.004), (250, 0.0008)] 
    else:
        lr_schedule = [(1, 0.1), (82, 0.01), (123, 0.001), (250, 0.0002)]
    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule)
        ],
        model=Model(NUM_UNITS,WIDTH,INIT_CHANNEL,NUM_CLASSES,weights),
        steps_per_epoch=steps_per_epoch,
        max_epoch=70,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, \
                              i.e., number of units per prediction',
                        type=int, default=NUM_UNITS_PER_STACK)
    parser.add_argument('--num_classes', help='Number of classes', 
                        type=int, default=NUM_CLASSES)
    parser.add_argument('--stopgrad', help='Whether to stop gradients.',
                        type=bool, default=STOP_GRADIENTS)
    parser.add_argument('--stopgradpartial', help='Whether to stop gradients for other width.',
                        type=bool, default=STOP_GRADIENTS_PARTIAL)
    parser.add_argument('--sg_gamma', help='Gamma for partial stop_gradient',
                        type=np.float32, default=SG_GAMMA)
    parser.add_argument('--samloss', help='Method to Sample losses to update',
                        type=int, default=SAMLOSS)
    parser.add_argument('--exp_gamma', help='Gamma for exp3 in sample loss',
                        type=np.float32, default=EXP3_GAMMA)
    parser.add_argument('--sum_rand_ratio', help='frac{Sum weight}{randomly selected weight}',
                        type=np.float32, default=SUM_RAND_RATIO)
    parser.add_argument('--track_grads', help='Whether to track gradient l2 of each loss',
                        type=bool, default=TRACK_GRADIENTS)
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=OPTIMAL_AT)
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=5)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    FUNC_TYPE = args.func_type
    SAMLOSS = args.samloss
    OPTIMAL_AT = args.opt_at
    BATCH_SIZE = args.batch_size
    NUM_UNITS = args.num_units
    WIDTH = args.width
    INIT_CHANNEL = args.init_channel
    NUM_UNITS_PER_STACK = args.stack
    NUM_CLASSES = 10 #SVHN is always 10 class
    STOP_GRADIENTS = args.stopgrad
    STOP_GRADIENTS_PARTIAL = args.stopgradpartial
    SG_GAMMA = args.sg_gamma
    EXP3_GAMMA = args.exp_gamma
    SUM_RAND_RATIO = args.sum_rand_ratio
    TRACK_GRADIENTS = args.track_grads

    if STOP_GRADIENTS:
        STOP_GRADIENTS_PARTIAL = True
        SG_GAMMA = 0.0
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir(log_root=args.log_dir)
    utils.set_dataset_path(path=args.data_dir, auto_download=False)

    logger.info("On Dataset SVHN, Parameters: f= {} num_classes= {} n= {}, w= {}, c= {}, s= {}, batch_size= {}, stopgrad= {}, stopgradpartial= {}, sg_gamma= {}, rand_loss_selector= {}, exp_gamma= {}, sum_rand_ratio= {} opt_at= {}".format(\
                FUNC_TYPE,
                NUM_CLASSES, NUM_UNITS, WIDTH, INIT_CHANNEL, \
                NUM_UNITS_PER_STACK, BATCH_SIZE, STOP_GRADIENTS, \
                STOP_GRADIENTS_PARTIAL, SG_GAMMA, \
                args.samloss, EXP3_GAMMA, SUM_RAND_RATIO, \
                OPTIMAL_AT))


    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
