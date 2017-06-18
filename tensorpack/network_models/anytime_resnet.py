import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import anytime_loss, logger, utils, fs
from tensorpack.callbacks import Exp3CPU, RWMCPU, FixedDistributionCPU, ThompsonSamplingCPU

from tensorflow.contrib.layers import variance_scaling_initializer

NUM_RES_BLOCKS = 3


def uniform_cfg(n_units_per_block, n_blocks=NUM_RES_BLOCKS):
    return ([n_units_per_block] * n_blocks, residual)

def parser_add_arguments(parser):
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=128)
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=5)
    parser.add_argument('-w', '--width',
                        help='width of the network',
                        type=int, default=1)
    parser.add_argument('-c', '--init_channel',
                        help='channel at beginning of each width of the network',
                        type=int, default=16)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, \
                              i.e., number of units per prediction',
                        type=int, default=1)
    parser.add_argument('--num_classes', help='Number of classes', 
                        type=int, default=10)
    parser.add_argument('--stop_gradient', help='Whether to stop gradients.',
                        type=bool, default=False)
    parser.add_argument('--sg_gamma', help='Gamma for partial stop_gradient',
                        type=np.float32, default=0)
    parser.add_argument('--samloss', 
                        help='Method to Sample losses to update',
                        type=int, default=0)
    parser.add_argument('--exp_gamma', help='Gamma for exp3 in sample loss',
                        type=np.float32, default=0.3)
    parser.add_argument('--sum_rand_ratio', help='frac{Sum weight}{randomly selected weight}',
                        type=np.float32, default=2.0)
    parser.add_argument('--last_reward_rate', help='rate of last reward in comparison to the max',
                        type=np.float32, default=0.85)
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=5)
    parser.add_argument('--exponential_base', help='Exponential base',
                        type=np.float32)
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=-1)
    return parser

class AnytimeResnet(ModelDesc):
    """
    """
    def __init__(self, n, width, init_channel, num_classes, 
                 weights, input_size, args):
        """
        """
        super(AnytimeResnet, self).__init__()
        self.n = n
        self.width = width
        self.init_channel = init_channel
        self.num_classes = num_classes
        self.weights = weights
        self.input_size = input_size
        self.select_idx_name = "select_idx"

        self.options = args
        self.options.ls_method = self.options.samloss

        self.total_units = AnytimeResnet.compute_total_units(NUM_RES_BLOCKS, 
            self.n, self.width)

    def _get_inputs(self):
        return [InputDesc(tf.float32, \
                    [None, self.input_size, self.input_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]
    
    @staticmethod
    def compute_total_units(n_blocks, n, width):
        return n_blocks * n * width

    def compute_scope_basename(self, layer_idx):
        return "layer{:03d}".format(layer_idx)

    def compute_classification_callbacks(self):
        vcs = []
        total_units = self.total_units
        unit_idx = -1
        layer_idx=-1
        for bi in range(NUM_RES_BLOCKS):
            for k in range(self.n):
                layer_idx += 1
                for wi in range(self.width):
                    unit_idx += 1
                    weight = self.weights[unit_idx]
                    if weight > 0:
                        scope_name = self.compute_scope_basename(layer_idx)
                        scope_name += '.'+str(wi)+'.pred/' 
                        vcs.append(ClassificationError(\
                            wrong_tensor_name=scope_name+'wrong-top1:0', 
                            summary_name=scope_name+'val_err'))
                        vcs.append(ClassificationError(\
                            wrong_tensor_name=scope_name+'wrong-top5:0', 
                            summary_name=scope_name+'val-err5'))
        return vcs

    def compute_loss_select_callbacks(self):
        if self.options.ls_method > 0:
            ls_K = np.sum(np.asarray(self.weights) > 0)
            reward_names = [ 'tower0/reward_{:02d}:0'.format(i) for i in range(ls_K)]
            select_idx_name = '{}:0'.format(self.select_idx_name)
            if self.options.ls_method == 3:
                online_learn_cb = FixedDistributionCPU(ls_K, select_idx_name, None)
            elif self.options.ls_method == 6:
                online_learn_cb = FixedDistributionCPU(ls_K, select_idx_name, 
                    self.weights[self.weights>0])
            else:    
                gamma = self.options.exp3_gamma
                if self.options.ls_method == 1:
                    online_learn_func = Exp3CPU
                    gamma = 1.0
                elif self.options.ls_method == 2:
                    online_learn_func = Exp3CPU
                elif self.options.ls_method == 4:
                    online_learn_func = RWMCPU
                elif self.options.ls_method == 5:
                    online_learn_func = ThompsonSamplingCPU
                online_learn_cb = online_learn_func(ls_K, gamma, 
                    select_idx_name, reward_names)
            online_learn_cbs = [ online_learn_cb ]
        else:
            online_learn_cbs = []
        return online_learn_cbs


    def residual(name, l_feats, increase_dim=False):
        """
        Basic residual function for WANN: for index w, 
        the input feat is the concat of all featus upto and including 
        l_feats[w]. The output should have the same dimension
        as l_feats[w] for each w, if increase_dim is False

        Residual unit contains two 3x3 conv. The input is added
        to the final result of the conv path. 
        The identity path has no bn/relu.
        The conv path has preact (bn); in between the two conv
        there is a bn_relu; the final conv is followed by bn. 

        Note the only relu is in between convs. (following pyramidial net) 

        When dim increases, the first conv has stride==2, and doubles
        the dimension. The final addition pads the new channels with zeros.

        """
        shape = l_feats[0].get_shape().as_list()
        in_channel = shape[1]

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
                    merged_feats = tf.concat([merged_feats, l], 1, name='concat_mf')
                l = Conv2D('conv1', merged_feats, out_channel, stride=stride1)
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
                    merged_feats = tf.concat([merged_feats, l], 1, name='concat_ef')
                ef = Conv2D('conv2', merged_feats, out_channel)
                # The second conv need to be BN before addition.
                ef = BatchNorm('bn2', ef)
                l = l_feats[w]
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0,0], [in_channel//2, in_channel//2], [0,0], [0,0]])
                ef += l
                l_end_feats.append(ef)
        return l_end_feats


    def residual_bottleneck(name, l_feats, ch_in_to_ch_base=4):
        """
        Bottleneck resnet unit for WANN. Input of index w, is 
        the concat of l_feats[0], ..., l_feats[w]. 

        The input to output has two paths. Identity paths has no activation. 
        If the dimensions of in/output mismatch, input is converted to 
        output dim via 1x1 conv with strides. 

        The input channel of each l_feat is ch_in; 
        the base channel is ch_base = ch_in // ch_in_to_ch_base.
        The conv paths contains three conv. 1x1, 3x3, 1x1. 
        The first two convs outputs have channel(depth) of ch_base. 
        The last conv has ch_base*4 output channels

        Within the same block, ch_in_to_ch_base is 4. 
        The first res unit has ch_in_to_ch_base 1. 
        The first res unit of other blocks has ch_in_to_ch_base 2. 

        ch_in_to_ch_base == 2 also triggers downsampling with stride 2 at 3x3 conv

        """
        assert ch_in_to_ch_base in [1,2,4], ch_in_to_ch_base
        ch_in = l_feats[0].get_shape().as_list()[1] 
        ch_base = ch_in // ch_in_to_ch_base

        stride=1
        if ch_in_to_ch_base == 2:
            # the first unit of block 2,3,4,... (1based)
            stride = 2

        l_new_feats = [] 
        for w in range(self.width): 
            with tf.variable_scope('{}.{}.0'.format(name, w)) as scope:
                l = BatchNorm('bn0', l_feats[w])
                if w == 0:
                    merged_feats = l
                else:
                    merged_feats = tf.concat([merged_feats, l], 1, name='concat') 
                l = (LinearWrap(merged_feats)
                    .Conv2D('conv1x1_0', ch_base, kernel_shape=1, nl=BNReLU)
                    .Conv2D('conv3x3_1', ch_base, stride=stride, nl=BNReLU) 
                    .Conv2D('conv1x1_2', ch_base*4, kernel_shape=1, nl=BatchNorm)())

                shortcut = l_feats[w]
                if ch_in_to_ch_base == 1:
                    # first residual unit of the first block
                    shortcut = Conv2D('conv_short', shortcut, ch_base*4, kernel_shape=1)
                elif ch_in_to_ch_base == 2:
                    # first residual unit of a block
                    shortcut = AvgPooling('pool', shortcut, stride)
                    shortcut = Conv2D('conv_short', shortcut, ch_base*4, kernel_shape=1)
                l = l + shortcut
                l_new_feats.append(l)
            # end var scope
        #end for
        return l_new_feats


    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.transpose(image, [0,3,1,2])

        def predictions_and_losses(name, l_feats, out_dim, label):
            l_logits = []
            var_list = []
            l_costs = []
            l_wrong = []
            l_wrong5 = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.pred') as scope:
                    l = tf.nn.relu(l_feats[w])
                    if self.options.prediction_1x1_conv:
                        ch_in = l.get_shape().as_list()[1]
                        l = Conv2D('conv1x1', l, ch_in, kernel_shape=1)
                        l = BNReLU('bnrelu1x1', l)
                    l = GlobalAvgPooling('gap', l)
                    if w == 0:
                        merged_feats = l
                    else:
                        merged_feats = tf.concat([merged_feats, l], 1, name='concat')
                    logits = FullyConnected('linear', merged_feats, out_dim, \
                                            nl=tf.identity)
                    var_list.append(logits.variables.W)
                    var_list.append(logits.variables.b)
                    #if w != 0:
                    #    logits += l_logits[-1]
                    l_logits.append(logits)

                    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                        logits=logits, labels=label)
                    cost = tf.reduce_mean(cost, name='cross_entropy_loss')
                    add_moving_summary(cost)

                    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
                    add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
                    
                    wrong5 = prediction_incorrect(logits, label, 5, name='wrong-top5')
                    add_moving_summary(tf.reduce_mean(wrong5, name='train-error-top5'))

                    l_costs.append(cost)
                    l_wrong.append(wrong)
                    l_wrong5.append(wrong5)

            return l_logits, var_list, l_costs, l_wrong, l_wrong5

        logger.info("sampling loss with method {}".format(self.options.ls_method))
        ls_K = np.sum(np.asarray(self.weights) > 0)
        if self.options.ls_method > 0:
            select_idx = tf.get_variable(self.select_idx_name, (), tf.int32,
                initializer=tf.constant_initializer(ls_K - 1), trainable=False)
            tf.summary.scalar(self.select_idx_name, select_idx)
            for i in range(ls_K):
                weight_i = tf.cast(tf.equal(select_idx, i), tf.float32, 
                                   name='weight_{:02d}'.format(i))
                add_moving_summary(weight_i)

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], 
                      data_format='NCHW'), \
             argscope(Conv2D, kernel_shape=3, nl=tf.identity, use_bias=False, 
                      W_init=variance_scaling_initializer(mode='FAN_OUT')):
            l_feats = [] 
            ll_feats = []
            for w in range(self.width):
                with tf.variable_scope('init_conv'+str(w)) as scope:
                    l = Conv2D('conv0', image, self.init_channel) 
                    #l = BatchNorm('bn0', l)
                    #l = tf.nn.relu(l)
                    l_feats.append(l)

            wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                              480000, 0.2, True)

            # Do not regularize for stop-gradient case, because
            # stop-grad requires cycling lr, and switching training targets
            if self.options.stop_gradient:
                wd_w = 0

            wd_cost = 0
            cost = 0
            unit_idx = -1
            layer_idx = -1
            anytime_idx = -1
            online_learn_rewards = []
            last_cost = None
            max_reward = 0.0
            for res_block_i in range(NUM_RES_BLOCKS):
                for k in range(self.n):
                    layer_idx += 1
                    scope_name = self.compute_scope_basename(layer_idx)
                    l_feats = residual(scope_name, l_feats, \
                                       increase_dim=(k==0 and res_block_i > 0))
                    ll_feats.append(l_feats)

                    # In case that we need to stop gradients
                    is_last_row = res_block_i == NUM_RES_BLOCKS-1 and k==self.n-1
                    if self.options.stop_gradient and not is_last_row:
                        l_new_feats = []
                        for fi, f in enumerate(l_feats):
                            unit_idx +=1
                            if self.weights[unit_idx] > 0:
                                f = (1-self.options.sg_gamma)*tf.stop_gradient(f) \
                                   + self.options.sg_gamma*f
                            l_new_feats.append(f)
                        l_feats = l_new_feats
                # end for each k in self.n
            #end for each block
            
            unit_idx = -1
            for layer_idx, l_feats in enumerate(ll_feats):
                scope_name = self.compute_scope_basename(layer_idx)
                l_logits, var_list, l_costs, l_wrong, l_wrong5 = \
                    predictions_and_losses(scope_name, l_feats, 
                                           self.num_classes, label)
                    
                for ci, c in enumerate(l_costs):
                    unit_idx += 1
                    cost_weight = self.weights[unit_idx]
                    if cost_weight > 0:
                        anytime_idx += 1

                        # Additional weight for unit_idx. 
                        add_weight = 0
                        if self.options.ls_method > 0:
                            add_weight = tf.cond(tf.equal(anytime_idx, 
                                                          select_idx),
                                lambda: tf.constant(self.weights[-1] * 2.0, 
                                                    dtype=tf.float32),
                                lambda: tf.constant(0, dtype=tf.float32))
                        if self.options.sum_rand_ratio > 0:
                            cost += (cost_weight + add_weight / \
                                self.options.sum_rand_ratio) * c
                        else:
                            cost += add_weight * c

                        # Regularize weights from FC layers.
                        wd_cost += cost_weight * wd_w * tf.nn.l2_loss(var_list[2*ci])
                        
                        ###############
                        # Compute reward for loss selecters. 

                        # Compute gradients of the loss as the rewards
                        #gs = tf.gradients(c, tf.trainable_variables()) 
                        #reward = tf.add_n([tf.nn.l2_loss(g) for g in gs if g is not None])
                        # Compute relative loss improvement as rewards
                        if not last_cost is None:
                            reward = 1.0 - c / last_cost
                            max_reward = tf.maximum(reward, max_reward)
                            online_learn_rewards.append(tf.multiply(reward, 1.0, 
                                name='reward_{:02d}'.format(anytime_idx-1)))
                        if ci == len(l_costs)-1 and is_last_row:
                            reward = max_reward * self.options.last_reward_rate
                            online_learn_rewards.append(tf.multiply(reward, 1.0, 
                                name='reward_{:02d}'.format(anytime_idx)))
                            #cost = tf.Print(cost, online_learn_rewards)
                        last_cost = c

                    #endif cost_weight > 0
                #endfor each width
            #endfor each layer
        #end argscope

        # weight decay on all W on conv layers for regularization
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), \
                         name='wd_cost')
        cost = tf.identity(cost, name='sum_losses')
        add_moving_summary(cost, wd_cost)
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt
