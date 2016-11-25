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
    def __init__(self, n, width, init_channel):
        super(AnytimeModel, self).__init__()
        self.n = n
        self.width = width
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

        def residual(name, l_feats, l_feats_prerelu=None, increase_dim=False):
            shape = l_feats[0].get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            if l_feats_prerelu is None:
                l_feats_prerelu = l_feats

            l_mid_feats = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.mid') as scope:
                    mf = 0 
                    for ww in range(w+1):
                        c1 = conv('conv1.'+str(ww), l_feats[ww], out_channel, stride1)
                        mf += c1
                    mf = BatchNorm('bn1', mf)
                    mf = tf.nn.relu(mf)
                    l_mid_feats.append(mf)

            l_end_feats_prerelu = []
            l_end_feats = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.end') as scope:
                    ef = 0
                    for ww in range(w+1):
                        c2 = conv('conv2.'+str(ww), l_mid_feats[ww], out_channel, 1)
                        ef += c2
                    ef = BatchNorm('bn2', ef)
                    l = l_feats[w]
                    if increase_dim:
                        l = AvgPooling('pool', l, 2)
                        l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])
                    ef += l
                    l_end_feats_prerelu.append(ef)
                    # Uncomment to turn on the final relu at each resnet block
                    #ef = tf.nn.relu(ef)
                    l_end_feats.append(ef)
            return l_end_feats, l_end_feats_prerelu

        def row_sum_predict(name, l_feats, out_dim):
            l_logits = []
            var_list = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.predict') as scope:
                    # If resnet last relu is active in resnet blocks, 
                    # we don't have to relu each feature before prediction. 
                    #l = l_feats[w]
                    l = tf.nn.relu(l_feats[w])
                    l = GlobalAvgPooling('gap', l)
                    logits, vl = FullyConnected('linear', l, out_dim, nl=tf.identity, return_vars=True)
                    var_list.extend(vl)
                    if w != 0:
                        logits += l_logits[-1]
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

                    wrong = prediction_incorrect(logits, label)
                    nr_wrong = tf.reduce_sum(wrong, name='wrong') # for testing
                    wrong = tf.reduce_mean(wrong, name='train_error')

                    l_costs.append(cost)
                    l_wrong.append(wrong)
            return l_costs, l_wrong


        def loss_weights(N):
            log_n = int(np.log2(N))
            weights = np.zeros(N)
            for j in range(log_n + 1):
                t = int(2**j)
                #wj = [ (1 + i // t)**(-2) if i%t==0 else 0 for i in range(N) ] 
                #wj = [ 0.7**(i//t) if i%t==0 else 0 for i in range(N) ] 
                #wj /= np.sum(wj)
                wj = [ 1 if i%t==0 else 0 for i in range(N) ] 
                weights += wj
            weights[0] = np.sum(weights[1:])
            #weights /= np.sum(weights)
            return weights
                

        l_feats = [] 
        total_channel = self.init_channel
        for w in range(self.width):
            with tf.variable_scope('init_conv'+str(w)) as scope:
                l = conv('conv0', image, total_channel//self.width, 1) 
                l = BatchNorm('bn0', l)
                l = tf.nn.relu(l)
                l_feats.append(l)
        l_feats_prerelu = l_feats

        # weight decay for all W of fc/conv layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = 0
        cost = 0
        node_rev_idx = NUM_RES_BLOCKS * self.n * self.width
        cost_weights = loss_weights(node_rev_idx) 
        for res_block_i in range(NUM_RES_BLOCKS):
            # {32, c_total=16}, {16, c=32}, {8, c=64}
            for k in range(self.n):
                scope_name = 'res{}.{}'.format(res_block_i, k)
                l_feats, l_feats_prerelu = residual(scope_name, l_feats, l_feats_prerelu, increase_dim=(k==0 and res_block_i > 0))
                l_logits, var_list = row_sum_predict(scope_name, l_feats, out_dim=10)
                l_costs, l_wrong = cost_and_eval(scope_name, l_logits, label)

                for ci, c in enumerate(l_costs):
                    cost_weight = cost_weights[node_rev_idx - 1]
                    # Uncomment to have weight only on the last layer
                    cost_weight = 1 if node_rev_idx == 9 else 0
                    #cost_weight = 0.26 if node_rev_idx == 13 else cost_weight
                    #cost_weight = 0.1 if node_rev_idx == 15 else cost_weight
                    #cost_weight = 0.7**(node_rev_idx-1)
                    cost += cost_weight * c
                    wd_cost += cost_weight * wd_w * tf.nn.l2_loss(var_list[2*ci])
                    node_rev_idx -= 1
                
                add_moving_summary(l_costs)
                add_moving_summary(l_wrong)

        # regularize conv
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')


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

    n=5
    width=1
    init_channel=16
    vcs = []
    for ri in range(NUM_RES_BLOCKS):
        for i in range(n):
            for w in range(width):
                scope_name = 'res{}.{}.{}.eval/'.format(ri, i, w)
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
        model=AnytimeModel(n=n, width=width, init_channel=init_channel),
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
