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

from tensorflow.contrib.layers import variance_scaling_initializer

"""
"""

BATCH_SIZE = 128
NUM_RES_BLOCKS = 3
NUM_UNITS = 5
WIDTH = 1
INIT_CHANNEL = 16

EXP_BASE=2.0
STOP_GRADIENTS=False

def loss_weights(N):
    return anytime_loss.exponential_weights(N, base=EXP_BASE)

class Model(ModelDesc):

    def __init__(self, n, width, init_channel):
        super(Model, self).__init__()
        self.n = n
        self.width = width
        self.init_channel = init_channel

    def _get_inputs(self):
        return [InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                InputVar(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0 - 1

        def conv(name, l, channel, stride):
            kernel = 3
            return Conv2D(name, l, channel, kernel, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/kernel/kernel/channel)))

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
                    if w == 0:
                        merged_feats = l_feats[0]
                    else:
                        merged_feats = tf.concat(3, [merged_feats, l_feats[w]], name='concat_mf')
                    mf = conv('conv1', merged_feats, out_channel, stride1)
                    mf = BatchNorm('bn1', mf)
                    mf = tf.nn.relu(mf)
                    l_mid_feats.append(mf)

            l_end_feats = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.end') as scope:
                    if w == 0:
                        merged_feats = l_mid_feats[0]
                    else: 
                        merged_feats = tf.concat(3, [merged_feats, l_mid_feats[w]], name='concat_ef')
                    ef = conv('conv2', merged_feats, out_channel, 1)
                    ef = BatchNorm('bn2', ef)
                    l = l_feats[w]
                    if increase_dim:
                        l = AvgPooling('pool', l, 2)
                        l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])
                    ef += l
                    # Uncomment to turn on the final relu at each resnet block
                    #ef = tf.nn.relu(ef)
                    l_end_feats.append(ef)
            return l_end_feats

        def row_sum_predict(name, l_feats, out_dim, is_last):
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
                    logits, vl = FullyConnected('linear', merged_feats, out_dim, nl=tf.identity, return_vars=True)
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
                l = BatchNorm('bn0', l)
                l = tf.nn.relu(l)
                l_feats.append(l)

        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = 0
        cost = 0
        total_units = NUM_RES_BLOCKS * self.n * self.width
        cost_weights = loss_weights(total_units)
        unit_idx = 0
        for res_block_i in range(NUM_RES_BLOCKS):
            for k in range(self.n):
                scope_name = 'res{}.{:02d}'.format(res_block_i, k)
                l_feats = \
                    residual(scope_name, l_feats, 
                             increase_dim=(k==0 and res_block_i > 0))
                l_logits, var_list = \
                    row_sum_predict(scope_name, l_feats, out_dim=10, 
                                    is_last= k==self.n-1 and res_block_i == NUM_RES_BLOCKS-1)
                l_costs, l_wrong = cost_and_eval(scope_name, l_logits, label)

                # Stop gradients from uppper layers. 
                if STOP_GRADIENTS:
                    l_feats = [tf.stop_gradient(feats) for feats in l_feats]

                for ci, c in enumerate(l_costs):
                    cost_weight = cost_weights[unit_idx]
                    unit_idx += 1
                    if cost_weight > 0:
                        cost += cost_weight * c
                        # regularize weights from FC layers
                        # Should use regularize_cost to get the weights using variable names
                        wd_cost += cost_weight * wd_w * tf.nn.l2_loss(var_list[2*ci])


        # weight decay on all W on conv layers
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
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


    lr = get_scalar_var('learning_rate', 0.01, summary=True)
    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost')] + vcs),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        model=Model(n=NUM_UNITS,width=WIDTH,init_channel=INIT_CHANNEL),
        steps_per_epoch=steps_per_epoch,
        max_epoch=400,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=BATCH_SIZE)
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=5)
    parser.add_argument('-w', '--width',
                        help='width of the network',
                        type=int, default=1)
    parser.add_argument('-c', '--init_channel',
                        help='channel at beginning of each width of the network',
                        type=int, default=16)
    parser.add_argument('-b', '--base', 
                        help='Exponential base',
                        type=np.float32, default=1)
    parser.add_argument('--stopgrad', help='Whether to stop gradients.',
                        type=bool, default=False)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    NUM_UNITS = args.num_units
    WIDTH = args.width
    INIT_CHANNEL = args.init_channel
    EXP_BASE = args.base
    STOP_GRADIENTS = args.stopgrad
    

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if os.getenv('LOG_DIR') is None:
        logger.auto_set_dir()
    else:
        logger.auto_set_dir(log_root = os.environ['LOG_DIR'])
    if os.getenv('DATA_DIR') is not None:
        os.environ['TENSORPACK_DATASET'] = os.environ['DATA_DIR']
    logger.info("Parameters: n= {}, w= {}, c= {}, b= {}, batch_size={}, stopgrad= {}".format(NUM_UNITS,\
        WIDTH, INIT_CHANNEL, EXP_BASE, BATCH_SIZE, STOP_GRADIENTS))

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
