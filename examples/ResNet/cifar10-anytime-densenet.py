#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
import tensorpack.utils.anytime_loss as anytime_loss

"""
"""

NUM_RES_BLOCKS=3
NUM_UNITS=12
GROWTH_RATE=12
INIT_CHANNEL=16

NUM_UNITS_PER_STACK=1

def loss_weights(N):
    return anytime_loss.stack_loss_weights(N, NUM_UNITS_PER_STACK)

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
        unit_idx = 0
        
        merged_feats = None
        for bi in range(NUM_RES_BLOCKS):
            if bi == 0:
                with tf.variable_scope('init_conv') as scope:
                    l = conv('conv', image, self.init_channel, 1) 
            else:
                with tf.variable_scope('trans_{}'.format(bi)) as scope:
                    l = merged_feats
                    l = BatchNorm('bn'.format(bi), l)
                    l = tf.nn.relu(l)
                    l = conv('conv', l, int(l.get_shape().as_list()[3] / self.reduction_rate), 1)
                    l = tf.nn.relu(l)
                    l = AvgPooling('pool', l, 2)
            merged_feats = l

            for k in range(self.n):
                cost_weight = cost_weights[unit_idx]
                unit_idx += 1
                scope_name = 'dense{}.{}'.format(bi, k)
                with tf.variable_scope(scope_name) as scope:
                    l = merged_feats
                    #l = BatchNorm('bn0', l)
                    #l = tf.nn.relu(l)
                    #l = conv('conv0', l, self.growth_rate * self.bottleneck_width, 1, 1)
                    l = BatchNorm('bn1', l)
                    l = tf.nn.relu(l)
                    l = conv('conv1', l, self.growth_rate, 1)

                    merged_feats = tf.concat(3, [merged_feats, l], name='concat')
                    if cost_weight >0:
                        #print "Stop gradient at {}".format(scope_name)
                        #merged_feats = tf.stop_gradient(merged_feats)
                        l = BatchNorm('bn_pred', l)
                        l = tf.nn.relu(l)
                        l = GlobalAvgPooling('gap', l)
                        logits, vl = FullyConnected('linear', l, out_dim=10, nl=tf.identity, return_vars=True)
                        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
                        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

                        wrong = prediction_incorrect(logits, label)
                        tra_err = tf.reduce_mean(wrong, name='train_error')

                        total_cost += cost_weight * cost
                        #wd_cost += cost_weight * wd_w * (tf.nn.l2_loss(vl[0]) + tf.nn.l2_loss(vl[1]))
                        #wd_cost += wd_w * regularize_cost('{}/conv/W'.format(scope_name), tf.nn.l2_loss)

                        print '{} {} {}'.format(bi, k, cost_weight)

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
    ds = dataset.Cifar10(train_or_test, shuffle=True)
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
    ds = BatchData(ds, 64, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    logger.auto_set_dir()

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
    unit_idx = 0
    for bi in range(NUM_RES_BLOCKS):
        for ui in range(NUM_UNITS):
            scope_name = 'dense{}.{}/'.format(bi, ui)
            if cost_weights[unit_idx] > 0:
                vcs.append(ClassificationError(\
                    wrong_tensor_name=scope_name+'incorrect_vector:0', 
                    summary_name=scope_name+'val_err'))
            unit_idx += 1

    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        callbacks=[
            StatPrinter(),
            ModelSaver(),
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
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=12)
    parser.add_argument('-g', '--growth_rate',
                        help='number of channel per new layer',
                        type=int, default=12)
    parser.add_argument('-c', '--init_channel',
                        help='number of initial channels',
                        type=int, default=16)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, i.e., number of units per prediction',
                        type=int, default=1)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    NUM_UNITS = args.num_units
    GROWTH_RATE = args.growth_rate
    INIT_CHANNEL = args.init_channel
    NUM_UNITS_PER_STACK = args.stack

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
        config.set_tower(tower=map(int, args.gpu.split(',')))
    SyncMultiGPUTrainer(config).train()
