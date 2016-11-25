#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027

I can reproduce the results on 2 TitanX for
n=5, about 7.1% val error after 67k step (8.6 step/s)
n=18, about 5.7% val error (2.45 step/s)
n=30: a 182-layer network, about 5.6% val error after 51k step (1.55 step/s)
This model uses the whole training set instead of a train-val split.
"""

NUM_RES_BLOCKS=3

class AnytimeModel(ModelDesc):
    def __init__(self, n, growth_rate, init_channel):
        super(AnytimeModel, self).__init__()
        self.n = n
        self.growth_rate = growth_rate
        self.init_channel = init_channel

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                InputVar(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1

        def conv(name, l, channel, stride):
            kernel = 3
            return Conv2D(name, l, channel, kernel, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/kernel/kernel/channel)))

        def loss_weights(N):
            log_n = int(np.log2(N))
            weights = np.zeros(N)
            for j in range(log_n + 1):
                t = int(2**j)
                wj = [ 1 if i%t==0 else 0 for i in range(N) ] 
                weights += wj
            weights[0] = np.sum(weights[1:])
            weights /= np.sum(weights)
            weights *= N
            return weights

        
        # weight decay for all W of fc/conv layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = 0
        total_cost = 0
        node_rev_idx = NUM_RES_BLOCKS * self.n
        cost_weights = loss_weights(node_rev_idx) 
        
        merged_feats = None
        for bi in range(NUM_RES_BLOCKS):
            if bi == 0:
                with tf.variable_scope('init_conv') as scope:
                    l = conv('conv0', image, self.init_channel, 1) 
                    #l = BatchNorm('bn0', l)
                    #l = tf.nn.relu(l)
            else:
                with tf.variable_scope('trans_{}'.format(bi)) as scope:
                    l = conv('conv1x1', merged_feats, merged_feats.get_shape().as_list()[3], 2)
                    #l = tf.nn.relu(l)
            merged_feats = l

            for k in range(self.n):
                scope_name = 'dense{}.{}'.format(bi, k)
                with tf.variable_scope(scope_name) as scope:
                    l = merged_feats
                    l = conv('conv', l, self.growth_rate, 1)
                    l = BatchNorm('bn', l)
                    l = tf.nn.relu(l)
                    feats = l
                    merged_feats = tf.concat(3, [merged_feats, feats], name='concat')
                    
                    l = GlobalAvgPooling('gap', l)
                    logits, vl = FullyConnected('linear', l, out_dim=10, nl=tf.identity, return_vars=True)
                    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
                    cost = tf.reduce_mean(cost, name='cross_entropy_loss')

                    wrong = prediction_incorrect(logits, label)
                    nr_wrong = tf.reduce_sum(wrong, name='wrong') # for testing
                    wrong = tf.reduce_mean(wrong, name='train_error')

                    cost_weight = cost_weights[node_rev_idx - 1]
                    total_cost += cost_weight * cost
                    wd_cost += cost_weight * wd_w * (tf.nn.l2_loss(vl[0]) + tf.nn.l2_loss(vl[1]))
                    wd_cost += cost_weight * wd_w * regularize_cost('{}/conv/W'.format(scope_name), tf.nn.l2_loss) / merged_feats.get_shape().as_list()[-1]
                    node_rev_idx -= 1

                    print '{} {} {}'.format(bi, k, cost_weight)

                    add_moving_summary(cost)
                    add_moving_summary(wrong)

        # regularize conv
        #wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), name='wd_cost')
        total_cost = tf.identity(total_cost, name='pred_cost')
        wd_cost = tf.identity(wd_cost, name='wd_cost')
        
        add_moving_summary(total_cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
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
            #imgaug.Brightness(20),
            #imgaug.Contrast((0.6,1.4)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    logger.auto_set_dir()

    # prepare dataset
    dataset_train = get_data('train')
    step_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    sess_config = get_default_sess_config(0.9)

    get_global_step_var()
    lr = tf.Variable(0.01, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    n=10
    growth_rate=12
    init_channel=16
    vcs = []
    for ri in range(NUM_RES_BLOCKS):
        for i in range(n):
            scope_name = 'dense{}.{}/'.format(ri, i)
            vcs.append(ClassificationError(wrong_var_name=scope_name+'wrong:0', summary_name=scope_name+'val_err'))

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ]),
        session_config=sess_config,
        model=AnytimeModel(n=n, growth_rate=growth_rate, init_channel=init_channel),
        step_per_epoch=step_per_epoch,
        max_epoch=400,
    )

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES=""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        pass

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
        config.set_tower(tower=map(int, args.gpu.split(',')))
    SyncMultiGPUTrainer(config).train()
