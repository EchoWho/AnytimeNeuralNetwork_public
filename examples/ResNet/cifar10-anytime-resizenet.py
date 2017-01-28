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
"""

NUM_RES_BLOCKS=3
BATCH_SIZE=64

def loss_weights(N):
    log_n = int(np.log2(N))
    weights = np.zeros(N)
    for j in range(log_n + 1):
        t = int(2**j) *2
        wj = [ 1 if i%t==0 else 0 for i in range(N) ] 
        weights += wj
    weights[0] = np.sum(weights[1:])
    weights /= np.sum(weights)
    weights *= log_n

    return weights

def conv_info(N, init_channel, init_size):
    size = init_size[0]
    final_size = size // 4
    size_change = final_size - size 

    channel = init_channel
    final_channel = init_channel * 4
    channel_change = final_channel - channel

    l_sizes = []
    l_channels = []
    for i in range(N):
        size = int(np.ceil(init_size[0] + size_change * (i+1) / N))
        l_sizes.append([size, size])

        channel = int(np.floor(init_channel + channel_change * (i+1) / N))
        l_channels.append(channel)
    
    return l_sizes, l_channels

class Model(ModelDesc):
    def __init__(self, n, growth_rate, init_channel):
        super(Model, self).__init__()
        self.n = n
        self.growth_rate = growth_rate
        self.init_channel = init_channel
        self.bottleneck_width = 4
        self.reduction_rate = 1

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                InputVar(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1

        def conv(name, l, channel, stride, kernel=3):
            return Conv2D(name, l, channel, kernel, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/kernel/kernel/channel)))

        def add_layer(scope, l_in, size, channel):
            l = BatchNorm('bn0', l_in)
            l = conv('conv1', l, channel, 1)
            l = BatchNorm('bn1', l)
            l = tf.nn.relu(l)
            l = conv('conv2', l, channel, 1)
            l = BatchNorm('bn2', l)
            
            channel_in = l_in.get_shape().as_list()[3] 
            l_in = tf.pad(l_in, [[0,0],[0,0],[0,0], [0, channel - channel_in]])

            l = l + l_in

            l = tf.image.resize_images(l, size, tf.image.ResizeMethod.BILINEAR, align_corners=False)
            return l

        wd_cost = 0
        total_cost = 0
        cost_weights = loss_weights(self.n) 
        l_sizes, l_channels = conv_info(self.n, self.init_channel, [32, 32])
        print l_sizes
        print l_channels

        with tf.variable_scope('init_conv') as scope:
            layer = conv('conv', image, self.init_channel, 1) 
        for li in range(self.n):
            scope_name = 'layer_{}'.format(li)
            with tf.variable_scope(scope_name) as scope:
                layer = add_layer(scope_name, layer, l_sizes[li], l_channels[li])
                
                cost_weight = cost_weights[self.n - li - 1]
                if cost_weight > 0:
                    l = BatchNorm('bn_pred', layer)
                    l = tf.nn.relu(l)
                    l = GlobalAvgPooling('gap', l)
                    logits, vl = FullyConnected('linear', l, out_dim=10, nl=tf.identity, return_vars=True)
                    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
                    cost = tf.reduce_mean(cost, name='cross_entropy_loss')

                    wrong = prediction_incorrect(logits, label)
                    nr_wrong = tf.reduce_sum(wrong, name='wrong') # for testing
                    tra_err = tf.reduce_mean(wrong, name='train_error')

                    total_cost += cost_weight * cost

                    print ' {} {}'.format(li, cost_weight)

                    add_moving_summary(cost)
                    add_moving_summary(tra_err)

        # regularize conv
        wd_cost = tf.mul(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        total_cost = tf.identity(total_cost, name='pred_cost')
        
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
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
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
    lr = tf.Variable(0.1, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    n=15
    growth_rate=12
    init_channel=16
    vcs = []
    cost_weights = loss_weights(n)
    for i in range(n):
        scope_name = 'layer_{}/'.format(i)
        if cost_weights[n - i - 1] > 0:
            vcs.append(ClassificationError(wrong_var_name=scope_name+'wrong:0', summary_name=scope_name+'val_err'))

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (90, 0.01), (125, 0.001), (250, 0.0002)])
        ]),
        session_config=sess_config,
        model=Model(n=n, growth_rate=growth_rate, init_channel=init_channel),
        step_per_epoch=step_per_epoch,
        max_epoch=350,
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
