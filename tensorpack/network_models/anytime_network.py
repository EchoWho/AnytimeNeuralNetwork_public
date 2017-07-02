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
from collections import namedtuple
import bisect


"""
Data format, and the resulting dimension for the channels.
('NCHW',1) or ('NHWC', 3)
"""
DATA_FORMAT='NCHW'
CHANNEL_DIM=1 if DATA_FORMAT == 'NCHW' else 3
HEIGHT_DIM=1 + int(DATA_FORMAT == 'NCHW')
WIDTH_DIM=2 + int(DATA_FORMAT == 'NCHW')


# Best choice for samloss for AANN if running anytime networks.
BEST_AANN_METHOD=6
# method id for not using AANN
NO_AANN_METHOD=0

# func type for computing optimal at options.opt_at see anytime_loss.loss_weights
FUNC_TYPE_OPT = 2
# func type for computing ANN/AANN
FUNC_TYPE_ANN = 5


"""
    cfg is a tuple that contains
    ([ <list of n_units per block], <b_type>, <start_type>)

    n_units_per_block is a list of int
    b_type is in ["basic", "bottleneck"]
    start_type is in ["basic", "imagenet"]
"""
NetworkConfig = namedtuple('Config', ['n_units_per_block', 'b_type', 's_type'])

def compute_cfg(options):
    if hasattr(options, 'depth') and options.depth is not None:
        if options.depth == 18:
            n_units_per_block = [2,2,2,2]
            b_type = 'basic'
        elif options.depth == 34:
            n_units_per_block = [3,4,6,3]
            b_type = 'basic'
        elif options.depth == 50:
            n_units_per_block = [3,4,6,3]
            b_type = 'bottleneck'
        elif options.depth == 101:
            n_units_per_block = [3,4,23,3]
            b_type = 'bottleneck'
        elif options.depth == 152:
            n_units_per_block = [3,8,36,3]
            b_type = 'bottleneck'
        else:
            raise ValueError('depth {} must be in [18, 34, 50, 101, 152]'\
                .format(options.depth))
        s_type = 'imagenet' 
        return NetworkConfig(n_units_per_block, b_type, s_type)

    elif hasattr(options, 'densenet_depth') and options.densenet_depth is not None:
        if options.densenet_depth == 121:
            n_units_per_block = [6, 12, 24, 16]
            #default_growth_rate = 32
        elif options.densenet_depth == 169:
            n_units_per_block = [6, 12, 32, 32]
            #default_growth_rate = 32
        elif options.densenet_depth == 201:
            n_units_per_block = [6, 12, 48, 32]
            #default_growth_rate = 32
        elif options.densenet_depth == 161:
            n_units_per_block = [6, 12, 36, 24]
            #default_growth_rate = 48
        else:
            raise ValueError('densenet depth {} is undefined'\
                .format(options.densenet_depth))
        b_type = 'bottleneck'
        s_type = 'imagenet'
        return NetworkConfig(n_units_per_block, b_type, s_type)#, default_growth_rate)

    elif hasattr(options, 'fcdense_depth') and options.fcdense_depth is not None:
        if options.fcdense_depth == 103:
            n_units_per_block = [ 4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4 ]
        else:
            raise ValueError('FC dense net depth {} is undefined'\
                .format(options.fcdense_depth))
        b_type = 'basic'
        s_type = 'basic'
        return NetworkConfig(n_units_per_block, b_type, s_type)

    elif options.num_units is not None: 
        #option.n is set
        return NetworkConfig([options.num_units]*options.n_blocks, 'basic', 'basic')

def compute_total_units(options, config=None):
    if config is None:
        config = compute_cfg(options)
    return sum(config.n_units_per_block) * options.width

def parser_add_common_arguments(parser):
    """
        Parser augmentation for anytime resnet/common 
    """
    # special group that handles the network depths
    # For each networ type, add its special arg name here I guess. 
    depth_group = parser.add_mutually_exclusive_group(required=True)
    depth_group.add_argument('-n', '--num_units',
                            help='number of units in each stage',
                            type=int)
    # network complexity
    parser.add_argument('--n_blocks', help='Number of residual blocks, don\'t change usually.'
                        +' Only used if num_units is set',
                        type=int, default=3)
    parser.add_argument('-w', '--width',
                        help='width of anytime network. usually set to 1 for memory issues',
                        type=int, default=1)
    parser.add_argument('-c', '--init_channel',
                        help='channel at beginning of each width of the network',
                        type=int, default=16)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, '
                        +'i.e., number of units per prediction, or prediction period',
                        type=int, default=1)
    parser.add_argument('--prediction_1x1_conv', 
                        help='whether use 1x1 before fc to predict',
                        default=False, action='store_true')

    # stop gradient / forward thinking
    parser.add_argument('--stop_gradient', help='Whether to stop gradients.',
                        default=False, action='store_true')
    parser.add_argument('--sg_gamma', help='Gamma for partial stop_gradient',
                        type=np.float32, default=0)

    # selecting loss 
    parser.add_argument('--init_select_idx', help='the loss anytime_idx to select initially',
                        type=int)
    parser.add_argument('--samloss', 
                        help='Method to Sample losses to update',
                        type=int, default=0)
    parser.add_argument('--exp_gamma', help='Gamma for exp3 in sample loss',
                        type=np.float32, default=0.3)
    parser.add_argument('--sum_rand_ratio', help='frac{Sum weight}{randomly selected weight}',
                        type=np.float32, default=2.0)
    parser.add_argument('--last_reward_rate', help='rate of last reward in comparison to the max',
                        type=np.float32, default=0.85)

    # loss_weight computation
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=FUNC_TYPE_ANN)
    parser.add_argument('--exponential_base', help='Exponential base',
                        type=np.float32)
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=-1)

    # misc
    parser.add_argument('--init_lr', help='The initial learning rate',
                        type=np.float32, default=0.01)
    parser.add_argument('--num_classes', help='Number of classes', 
                        type=int, default=10)
    parser.add_argument('--regularize_coef', help='How coefficient of regularization decay',
                        type=str, default='const', choices=['const', 'decay']) 
    return parser, depth_group


################################################
# Resnet 
################################################
def parser_add_resnet_arguments(parser):
    parser, depth_group = parser_add_common_arguments(parser)
    depth_group.add_argument('-d', '--depth',
                            help='depth of the network in number of conv',
                            type=int)
    return parser

class AnytimeNetwork(ModelDesc):
    def __init__(self, input_size, args):
        super(AnytimeNetwork, self).__init__()
        self.options = args
        self.input_size = input_size
        self.network_config = compute_cfg(self.options)
        self.total_units = compute_total_units(self.options, self.network_config)

        # Warn user if they are using imagenet but doesn't have the right channel
        self.init_channel = args.init_channel
        if self.network_config.s_type == 'imagenet' and self.init_channel != 64:
            logger.warn('Resnet imagenet requires 64 initial channels')

        self.n_blocks = len(self.network_config.n_units_per_block) 
        self.width = args.width
        self.num_classes = args.num_classes

        self.weights = anytime_loss.loss_weights(self.total_units, args)
        self.ls_K = np.sum(np.asarray(self.weights) > 0)
        logger.info('weights: {}'.format(self.weights))

        # special names and conditions
        self.select_idx_name = "select_idx"
        self.options.ls_method = self.options.samloss
        # (UGLY) due to the history of development. 1,...,5 requires rewards
        self.options.require_rewards = self.options.samloss < 6 and \
            self.options.samloss > 0

        if self.options.func_type == FUNC_TYPE_OPT \
            and self.options.ls_method != NO_AANN_METHOD:
            # special case: if we are computing optimal, don't do AANN
            logger.warn("Computing optimal requires not running AANN."\
                +" Setting samloss to be {}".format(NO_AANN_METHOD))
            self.options.ls_method = NO_AANN_METHOD
            self.options.samloss = NO_AANN_METHOD
    
    def _get_inputs(self):
        return [InputDesc(tf.float32, \
                    [None, self.input_size, self.input_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def compute_scope_basename(self, layer_idx):
        return "layer{:03d}".format(layer_idx)

    ###
    #   Important annoyance alert:
    #   Since this method is called typically before the _build_graph is called,
    #   we cannot know the var/tensor names dynamically during cb construction.
    #   Hence it's up to implementation to make sure the right names are used.
    #
    #   To fix this, we need _build_graph to know it's in test mode and construct
    #   right initialization, cbs, and surpress certain cbs/summarys. 
    #   (TODO)
    def compute_classification_callbacks(self):
        """
        """
        vcs = []
        total_units = self.total_units
        unit_idx = -1
        layer_idx=-1
        for n_units in self.network_config.n_units_per_block:
            for k in range(n_units):
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
            reward_names = [ 'tower0/reward_{:02d}:0'.format(i) for i in range(self.ls_K)]
            select_idx_name = '{}:0'.format(self.select_idx_name)
            if self.options.ls_method == 3:
                online_learn_cb = FixedDistributionCPU(self.ls_K, select_idx_name, None)
            elif self.options.ls_method == 6:
                online_learn_cb = FixedDistributionCPU(self.ls_K, select_idx_name, 
                    self.weights[self.weights>0])
            elif self.options.ls_method == 1000:
                # custom schedule. select_idx will be initiated for use.
                # set the cb to be None to force use to give 
                # a custom schedule/selector cb
                online_learn_cb = None
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
                online_learn_cb = online_learn_func(self.ls_K, gamma, 
                    select_idx_name, reward_names)
            online_learn_cbs = [ online_learn_cb ]
        else:
            online_learn_cbs = []
        return online_learn_cbs

    def _compute_init_l_feats(self, image):
        l_feats = []
        for w in range(self.width):
            with tf.variable_scope('init_conv'+str(w)) as scope:
                if self.network_config.s_type == 'basic':
                    l = Conv2D('conv0', image, self.init_channel, 3) #, nl=BNReLU) 
                else:
                    assert self.network_config.s_type == 'imagenet'
                    l = (LinearWrap(image)
                        .Conv2D('conv0', self.init_channel, 7, stride=2, nl=BNReLU)
                        .MaxPooling('pool0', shape=3, stride=2, padding='SAME')())
                l_feats.append(l)
        return l_feats
    
    def _compute_ll_feats(self, image):
        raise Exception("Invoked abstract AnytimeNetwork. Use a specific one instead")

    def _compute_prediction_and_loss(self, l, label, unit_idx):
        """
            l: feat_map of DATA_FORMAT 
            label: target to determine the loss
            unit_idx : the feature computation unit index.
        """
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, self.num_classes, nl=tf.identity)
            
        ## local cost/error_rate
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        add_moving_summary(cost)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
        
        wrong5 = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong5, name='train-error-top5'))
        return logits, cost

    def _preprocess_inputs(self, inputs):
        image, label = inputs
        if DATA_FORMAT == 'NCHW':
            image = tf.transpose(image, [0,3,1,2])
        return image, label

    def _build_graph(self, inputs):
        logger.info("sampling loss with method {}".format(self.options.ls_method))
        select_idx = self._get_select_idx()
        
        with argscope([Conv2D, Deconv2D, AvgPooling, MaxPooling, BatchNorm, GlobalAvgPooling], 
                      data_format=DATA_FORMAT), \
            argscope([Conv2D, Deconv2D], nl=tf.identity, use_bias=False, 
                     W_init=variance_scaling_initializer(mode='FAN_OUT')):

            image, label = self._preprocess_inputs(inputs)
            dynamic_batch_size = tf.identity(tf.shape(image)[0], name='dynamic_batch_size')
            ll_feats = self._compute_ll_feats(image)
            
            if self.options.stop_gradient:
                # NOTE:
                # Do not regularize for stop-gradient case, because
                # stop-grad requires cycling lr, and switching training targets
                wd_w = 0
            elif self.options.regularize_coef == 'const':
                wd_w = 1e-4
            elif self.options.regularize_coef == 'decay':
                wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                                  480000, 0.2, True)
            
            wd_cost = 0
            total_cost = 0
            unit_idx = -1
            anytime_idx = -1
            last_cost = None
            max_reward = 0.0
            online_learn_rewards = []
            for layer_idx, l_feats in enumerate(ll_feats):
                scope_name = self.compute_scope_basename(layer_idx)
                for w in range(self.width):
                    unit_idx += 1
                    cost_weight = self.weights[unit_idx]
                    if cost_weight == 0:
                        continue
                    ## cost_weight is implied to be >0
                    anytime_idx += 1
                    with tf.variable_scope(scope_name+'.'+str(w)+'.pred') as scope:
                        ## compute logit using features from layer layer_idx
                        l = tf.nn.relu(l_feats[w])
                        if w == 0:
                            merged_feats = l
                        else:
                            merged_feats = tf.concat([merged_feats, l], CHANNEL_DIM, name='concat')
                        l = merged_feats
                        if self.options.prediction_1x1_conv:
                            ch_in = l.get_shape().as_list()[CHANNEL_DIM]
                            l = Conv2D('conv1x1', l, ch_in, 1)
                            l = BNReLU('bnrelu1x1', l)
                        
                        logits, cost = self._compute_prediction_and_loss(l, label, unit_idx)
                    #end scope of layer.w.pred

                    ## Compute the contribution of the cost to total cost
                    # Additional weight for unit_idx. 
                    add_weight = 0
                    if select_idx is not None:
                        add_weight = tf.cond(tf.equal(anytime_idx, 
                                                      select_idx),
                            lambda: tf.constant(self.weights[-1] * 2.0, 
                                                dtype=tf.float32),
                            lambda: tf.constant(0, dtype=tf.float32))
                    if self.options.sum_rand_ratio > 0:
                        total_cost += (cost_weight + add_weight / \
                            self.options.sum_rand_ratio) * cost
                    else:
                        total_cost += add_weight * cost

                    ## Regularize weights from FC layers.
                    wd_cost += cost_weight * wd_w * tf.nn.l2_loss(logits.variables.W)

                    ## Compute reward for loss selecters. 

                    # Compute gradients of the loss as the rewards
                    #gs = tf.gradients(c, tf.trainable_variables()) 
                    #reward = tf.add_n([tf.nn.l2_loss(g) for g in gs if g is not None])

                    # Compute relative loss improvement as rewards
                    # note the rewards are outside varscopes.
                    if self.options.require_rewards:
                        if not last_cost is None:
                            reward = 1.0 - cost / last_cost
                            max_reward = tf.maximum(reward, max_reward)
                            online_learn_rewards.append(tf.multiply(reward, 1.0, 
                                name='reward_{:02d}'.format(anytime_idx-1)))
                        if anytime_idx == self.ls_K - 1:
                            reward = max_reward * self.options.last_reward_rate
                            online_learn_rewards.append(tf.multiply(reward, 1.0, 
                                name='reward_{:02d}'.format(anytime_idx)))
                            #cost = tf.Print(cost, online_learn_rewards)
                        last_cost = cost
                    #end if compute_rewards
                    #end (implied) if cost_weight > 0
                #endfor each width
            #endfor each layer
        #end argscope

        # weight decay on all W on conv layers for regularization
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss), \
                         name='wd_cost')
        total_cost = tf.identity(total_cost, name='sum_losses')
        add_moving_summary(total_cost, wd_cost)
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([total_cost, wd_cost], name='cost') # specify training loss

    def _get_optimizer(self):
        assert self.options.init_lr > 0, self.options.init_lr
        lr = get_scalar_var('learning_rate', self.options.init_lr, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

    def _get_select_idx(self):
        select_idx = None
        if self.options.ls_method > 0:
            init_idx = self.options.init_select_idx
            if init_idx is None:
                init_idx = self.ls_K - 1
            elif init_idx < 0:
                init_idx = self.ls_K + init_idx
            assert (init_idx >=0 and init_idx < self.ls_K), init_idx
            select_idx = tf.get_variable(self.select_idx_name, (), tf.int32,
                initializer=tf.constant_initializer(init_idx), trainable=False)
            tf.summary.scalar(self.select_idx_name, select_idx)
            for i in range(self.ls_K):
                weight_i = tf.cast(tf.equal(select_idx, i), tf.float32, 
                                   name='weight_{:02d}'.format(i))
                add_moving_summary(weight_i)
        return select_idx


class AnytimeResnet(AnytimeNetwork):
    def __init__(self, input_size, args):
        super(AnytimeResnet, self).__init__(input_size, args)

    def residual(self, name, l_feats, increase_dim=False):
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
        in_channel = shape[CHANNEL_DIM]

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
                    merged_feats = tf.concat([merged_feats, l], CHANNEL_DIM, name='concat_mf')
                l = Conv2D('conv1', merged_feats, out_channel, 3, stride=stride1)
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
                    merged_feats = tf.concat([merged_feats, l], CHANNEL_DIM, name='concat_ef')
                ef = Conv2D('conv2', merged_feats, out_channel, 3)
                # The second conv need to be BN before addition.
                ef = BatchNorm('bn2', ef)
                l = l_feats[w]
                if increase_dim:
                    l = AvgPooling('pool', l, shape=2, stride=2)
                    pad_paddings = [[0,0], [0,0], [0,0], [0,0]]
                    pad_paddings[CHANNEL_DIM] = [ in_channel//2, in_channel//2 ]
                    l = tf.pad(l, pad_paddings)
                ef += l
                l_end_feats.append(ef)
        return l_end_feats

    def residual_bottleneck(self, name, l_feats, ch_in_to_ch_base=4):
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
        ch_in = l_feats[0].get_shape().as_list()[CHANNEL_DIM] 
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
                    merged_feats = tf.concat([merged_feats, l], CHANNEL_DIM, name='concat') 
                l = (LinearWrap(merged_feats)
                    .Conv2D('conv1x1_0', ch_base, 1, nl=BNReLU)
                    .Conv2D('conv3x3_1', ch_base, 3, stride=stride, nl=BNReLU)
                    .Conv2D('conv1x1_2', ch_base*4, 1)())
                l = BatchNorm('bn_3', l)

                shortcut = l_feats[w]
                if ch_in_to_ch_base < 4:
                    shortcut = Conv2D('conv_short', shortcut, ch_base*4, \
                                      1, stride=stride)
                    shortcut = BatchNorm('bn_short', shortcut)
                l = l + shortcut
                l_new_feats.append(l)
            # end var scope
        #end for
        return l_new_feats

    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)

        ll_feats = []
        unit_idx = -1
        layer_idx = -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            for k in range(n_units):
                layer_idx += 1
                scope_name = self.compute_scope_basename(layer_idx)
                if self.network_config.b_type == 'basic':
                    l_feats = self.residual(scope_name, l_feats, \
                        increase_dim=(k==0 and bi > 0))
                else:
                    assert self.network_config.b_type == 'bottleneck'
                    ch_in_to_ch_base = 4
                    if k == 0:
                        ch_in_to_ch_base = 2
                        if bi == 0:
                            ch_in_to_ch_base = 1
                    l_feats = self.residual_bottleneck(scope_name, l_feats, \
                        ch_in_to_ch_base)
                ll_feats.append(l_feats)

                # In case that we need to stop gradients
                is_last_row = bi==self.n_blocks-1 and k==n_units-1
                if self.options.stop_gradient and not is_last_row:
                    l_new_feats = []
                    for fi, f in enumerate(l_feats):
                        unit_idx +=1
                        if self.weights[unit_idx] > 0:
                            f = (1-self.options.sg_gamma)*tf.stop_gradient(f) \
                               + self.options.sg_gamma*f
                            logger.info("stop gradient after unit {}".format(unit_idx))
                        l_new_feats.append(f)
                    l_feats = l_new_feats
            # end for each k in n_units
        #end for each block
        return ll_feats


################################################
# Dense Net (Log Dense)
################################################

def parser_add_densenet_arguments(parser):
    parser, depth_group = parser_add_common_arguments(parser)
    depth_group.add_argument('--densenet_depth',
                             help='depth of densenet for predefined networks',
                             type=int)
    parser.add_argument('-g', '--growth_rate', help='growth rate k for log dense',
                        type=int, default=16)
    parser.add_argument('--use_init_ch', 
        help='whether to use specified init channel argument, '\
        +' useful for networks that has init_ch based on'\
        +' other metrics such as densenet',
                        default=False, action='store_true')
    parser.add_argument('--dense_select_method', help='densenet previous feature selection choice',
                        type=int, default=0)
    parser.add_argument('--log_dense_coef', help='The constant multiplier of log(depth) to connect',
                        type=np.float32, default=1)
    parser.add_argument('--reduction_ratio', help='reduction ratio at transitions',
                        type=np.float32, default=1)
    return parser, depth_group


class AnytimeDensenet(AnytimeNetwork):
    def __init__(self, input_size, args):
        super(AnytimeDensenet, self).__init__(input_size, args)
        self.dense_select_method = self.options.dense_select_method
        self.log_dense_coef = self.options.log_dense_coef
        self.reduction_ratio = self.options.reduction_ratio
        self.growth_rate = self.options.growth_rate

        if not self.options.use_init_ch:
            default_ch = self.growth_rate * 2
            if self.init_channel != default_ch:
                self.init_channel = default_ch
                logger.info("Densenet sets the init_channel to be "
                    + "2*growth_rate by default. "
                    + "I'm setting this automatically!")
        
        # width > 1 is not implemented for densenet
        assert self.width == 1,self.width

        if self.options.func_type == FUNC_TYPE_ANN \
            and self.options.ls_method != BEST_AANN_METHOD:
            logger.warn("Densenet prefers using AANN instead of other methods."\
                +"Changing samloss to be {}".format(BEST_AANN_METHOD))
            self.options.ls_method = BEST_AANN_METHOD
            self.options.samloss = BEST_AANN_METHOD


    def dense_select_indices(self, ui):
        """
            ui : unit_idx
        """
        #if ui == self.total_units - 1:
        #    # special cast for the last, which selects all
        #    return list(range(self.total_units))

        if self.dense_select_method == 0:
            if not hasattr(self, 'exponential_diffs'):
                df = 1
                exponential_diffs = [0]
                while True:
                    df = df * 2
                    if df > self.total_units * self.log_dense_coef:
                        break
                    int_df = int((df-1) / self.log_dense_coef)
                    if int_df != exponential_diffs[-1]:
                        exponential_diffs.append(int_df)
                self.exponential_diffs = exponential_diffs
            diffs = self.exponential_diffs 
        elif self.dense_select_method == 1:
            diffs = list(range(int(np.log2(ui + 1) * self.log_dense_coef) + 1))
        elif self.dense_select_method == 2:
            n_select = int((np.log2(self.total_units +1) + 1) * self.log_dense_coef)
            diffs = list(range(int(np.log2(self.total_units + 1)) + 1))
        elif self.dense_select_method == 3:
            n_select = int((np.log2(self.total_units +1) + 1) * self.log_dense_coef)
            delta = (ui+1.0) / n_select
            df = 0
            diffs = []
            for i in range(n_select):
                int_df = int(df)
                if len(diffs) == 0 or int_df != diffs[-1]:
                    diffs.append(int_df)
                df += delta
        elif self.dense_select_method == 4:
            diffs = list(range(ui+1))
        indices = [ui - df  for df in diffs if ui - df >= 0 ]
        return indices

    def compute_block(self, pls, pmls, n_units):
        """
            pls : previous layers. including the init_feat. Hence pls[i] is from 
                layer i-1 for i > 0
            pmls : previous merged layers. (used for generate ll_feats) 
            n_units : num units in a block

            return pls, pmpls (updated version of these)
        """
        unit_idx = len(pls) - 2 # unit idx of the last completed unit
        for _ in range(n_units):
            unit_idx += 1
            scope_name = self.compute_scope_basename(unit_idx)
            with tf.variable_scope(scope_name+'.feat'):
                sl_indices = self.dense_select_indices(unit_idx)
                logger.info("unit_idx = {}, len past_feats = {}, selected_feats: {}".format(\
                    unit_idx, len(pls), sl_indices))
                
                ## Question: TODO whether save the pre-bnrelu feat or after-bnrelu feat
                # The current version saves the after-bnrelue to save mem/computation
                ml = tf.concat([pls[sli] for sli in sl_indices], \
                               CHANNEL_DIM, name='concat_feat')
                #ml = BNReLU('bnrelu_merged', ml)
                if self.network_config.b_type == 'bottleneck':
                    l = (LinearWrap(ml)
                        .Conv2D('conv1x1', 4*self.growth_rate, 1, nl=BNReLU)
                        .Conv2D('conv3x3', self.growth_rate, 3, nl=BNReLU)())
                else:
                    assert self.network_config.b_type == 'basic'
                    l = Conv2D('conv3x3', ml, self.growth_rate, 3, nl=BNReLU)
                #l = Dropout(l, keep_prob=0.2)
                pls.append(l)

                # If the feature is used for prediction, store it.
                if self.weights[unit_idx] > 0:
                    #l = BNReLU('bnrlu_local', l)
                    pmls.append(tf.concat([ml, l], CHANNEL_DIM, name='concat_pred'))
                else:
                    pmls.append(None)
        return pls, pmls

    def compute_transition(self, pls, trans_idx):
        """
            Note that this is not the same as the original densenet, 
            which 1x1 conv the whole merged feats. Here we 1x1 each 
            layer individually to prevent the mix.
        """
        new_pls = []
        for pli, pl in enumerate(pls):
            with tf.variable_scope('transit_{:02d}_{:02d}'.format(trans_idx, pli)): 
                ch_in = pl.get_shape().as_list()[CHANNEL_DIM]
                new_pls.append((LinearWrap(pl)
                    .Conv2D('conv', int(ch_in * self.reduction_ratio), 1, nl=BNReLU)
                    .AvgPooling('pool', 2, padding='SAME')()))
                # Dropout(l, keep_prob=0.2)
        return new_pls

    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        pls = [l_feats[0]]
        pmls = []
        for bi, n_units in enumerate(self.network_config.n_units_per_block):  
            pls, pmls = self.compute_block(pls, pmls, n_units)
            if bi != self.n_blocks - 1: 
                pls = self.compute_transition(pls, bi)
        
        ll_feats = [ [ feat ] for feat in pmls ]
        assert len(ll_feats) == self.total_units
        return ll_feats



################################################
# FCN for semantic labeling
#
# Important alert:
#   Since the layer H/W are induced based on 
#   the input layer rn, DO NOT give inputs of 
#   H/W that are not divisible by 2**n_pools.
#   Otherwise the upsampling results in H/W that
#   mismatch the input H/W. (And screw dynamic
#   shape checking in tensorflow....)
################################################
class AnytimeFCN(AnytimeNetwork):
    """
        Overload AnytimeNetwork from classification set-up to semantic labeling.
    """
    def __init__(self, args):
        super(AnytimeFCN, self).__init__(None, args)
                
    def compute_classification_callbacks(self):
        vcs = []
        total_units = self.total_units
        unit_idx = -1
        layer_idx=-1
        for n_units in self.network_config.n_units_per_block:
            for k in range(n_units):
                layer_idx += 1
                for wi in range(self.width):
                    unit_idx += 1
                    weight = self.weights[unit_idx]
                    if weight > 0:
                        scope_name = self.compute_scope_basename(layer_idx)
                        scope_name += '.'+str(wi)+'.pred/' 
                        vcs.append(MeanIoUFromConfusionMatrix(\
                            cm_name=scope_name+'confusion_matrix/SparseTensorDenseAdd:0', 
                            scope_name_prefix=scope_name+'val_'))
                        vcs.append(WeightedTensorStats(\
                            names=[scope_name+'sum_abs_diff:0', 
                                scope_name+'cross_entropy_loss:0'],
                            weight_name='dynamic_batch_size:0',
                            prefix='val_'))
        return vcs
        
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, None, 3], 'input'),
                InputDesc(tf.int32, [None, None, None], 'label')]

    def _preprocess_inputs(self, inputs):
        image, label_img = inputs
        if DATA_FORMAT == 'NCHW':
            image = tf.transpose(image, [0,3,1,2])

        eval_mask = tf.reshape(tf.less(label_img, self.num_classes), [-1], name='eval_mask')
        

        # NOTE:
        # If there are void labels, the one_hot generates the 0 distribution instead
        #
        # default dtype is tf.float32
        label_img = tf.one_hot(label_img, self.num_classes, axis=-1)

        # pre-compute label_img at various scales that the algorithm may need.
        l_label = [tf.reshape(label_img, [-1, self.num_classes], name='label_init')]
        # TODO not supported yet because it messes up the label_img_idx. 
        # Is there any FCN that uses imagenet start with double pooling?
        #if self.network_config.s_type == 'imagenet'
        #    label_img = AvgPooling('label_init_pool', label_img, 4, \
        #                           padding='SAME', data_format='NHWC')

        for pi in range(self.n_pools):
            label_img = AvgPooling('label_pool{}'.format(pi), label_img, 2, \
                                   padding='SAME', data_format='NHWC')
            label = tf.reshape(label_img, [-1, self.num_classes], \
                               name='label_{}'.format(pi))
            l_label.append(label)
        return image, (l_label, eval_mask)

    def _compute_prediction_and_loss(self, l, label_inputs, unit_idx):
        l_label, eval_mask = label_inputs
        # All previous layers have gone through BNReLU, so conv directly
        l = Conv2D('linear', l, self.num_classes, 1)
        logit_vars = l.variables
        if DATA_FORMAT == 'NCHW':
            l = tf.transpose(l, [0,2,3,1]) 
        logits = tf.reshape(l, [-1, self.num_classes])
        logits.variables = logit_vars

        # compute  block idx
        layer_idx = unit_idx // self.width
        csum = np.cumsum(self.network_config.n_units_per_block)
        # first idx that is > layer_idx
        bi = bisect.bisect_right(csum, layer_idx)

        # Case downsample check: n_pool uses label_idx=n_pool
        #   0 uses 0
        # Case upsample check: bi == n_pools uses label_idx=n_pools. 
        #   the final bi == n_pools * 2 uses 0
        label_img_idx = bi
        if bi > self.n_pools:
            label_img_idx = 2 * self.n_pools - bi

        label = l_label[label_img_idx] # note this is a probability of label distri

        # NOTE If it's not the last one, do not use eval_mask for confusion mat,
        # because some superpixels have partial void pixels and need to be trained.
        if not(label_img_idx == 0 and bi > 0):
            eval_mask = None
        else:
            float_mask = tf.cast(eval_mask, dtype=tf.float32)
            n_non_void_samples = tf.reduce_sum(float_mask) + 1e-20
            
        ## Local cost/loss for training

        # Have to implement our own weighted softmax cross entroy
        # because TF doesn't provide one 
        # Because logits and cost are returned in the end of this func, 
        # we use _logit to represent  the shifted logits.
        max_logits = tf.reduce_max(logits, axis=1, keep_dims=True)
        _logits = logits - max_logits
        normalizers = tf.reduce_sum(tf.exp(_logits), axis=1, keep_dims=True)
        _logits = _logits - tf.log(normalizers)
        cross_entropy = -tf.reduce_sum(\
            tf.multiply(label * _logits, self.class_weight), axis=1)
        if eval_mask is not None:
            cross_entropy = cross_entropy * float_mask
            cost = tf.divide(tf.reduce_sum(cross_entropy), n_non_void_samples,
                name='cross_entropy_loss')
        else:
            cost = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        add_moving_summary(cost)

        ## Other evaluation metrics 

        # total abs diff
        prob = tf.nn.softmax(logits, name='pred_prob')
        sum_abs_diff = sum_absolute_difference(prob, label)
        if eval_mask is not None:
            sum_abs_diff *= float_mask 
            sum_abs_diff = tf.divide(tf.reduce_sum(sum_abs_diff), 
                n_non_void_samples, name='sum_abs_diff')
        else:
            sum_abs_diff = tf.reduce_mean(sum_abs_diff, name='sum_abs_diff')
        add_moving_summary(sum_abs_diff)
        
        # confusion matrix for iou and pixel level accuracy
        int_pred = tf.argmax(logits, 1, name='int_pred')
        int_label = tf.argmax(label, 1, name='int_label')
        cm = tf.confusion_matrix(labels=int_label, predictions=int_pred,\
            num_classes=self.num_classes, name='confusion_matrix', weights=eval_mask)

        # pixel level accuracy
        accu = tf.cast(tf.reduce_sum(tf.diag_part(cm)), dtype=tf.float32) \
            / tf.cast(tf.reduce_sum(cm), dtype=tf.float32)
        accu = tf.identity(accu, name='accuracy')
        add_moving_summary(accu)

        return logits, cost


###########
## FCDense
def parser_add_fcdense_arguments(parser):
    parser, depth_group = parser_add_densenet_arguments(parser)
    depth_group.add_argument('--fcdense_depth',
                            help='depth of the network in number of conv',
                            type=int)
    parser.add_argument('--n_pools',
                        help='number of pooling blocks on the cfg.n_units_per_block',
                        type=int, default=None)
    return parser, depth_group


class FCDensenet(AnytimeFCN):
    def __init__(self, args):
        super(FCDensenet, self).__init__(args)
        self.reduction_ratio = self.options.reduction_ratio
        self.growth_rate = self.options.growth_rate

        # Class weight for fully convolutional networks
        self.class_weight = None
        if hasattr(args, 'class_weight'):
            self.class_weight = args.class_weight
        if self.class_weight is None:
            self.class_weight = np.ones(self.num_classes, dtype=np.float32) 
        logger.info('Class weights: {}'.format(self.class_weight))
        
        # FC-dense specific
        self.n_pools = args.n_pools
        
        # other format is not supported yet
        assert self.n_pools * 2 + 1 == self.n_blocks

        # FC-dense doesn't support width > 1 yet
        assert self.width == 1
        # FC-dense doesn't like the starting pooling of imagenet initial conv/pool
        assert self.network_config.s_type == 'basic'

        # This version doesn't support anytime prediction (yet) 
        assert self.options.func_type == FUNC_TYPE_OPT
    
    def _transition_up(self, skip_stack, l_layers, bi):
        with tf.variable_scope('TU_{}'.format(bi)) as scope:
            stack = tf.concat(l_layers, CHANNEL_DIM, name='concat_recent')
            ch_out = stack.get_shape().as_list()[CHANNEL_DIM]
            stack = Deconv2D('deconv', stack, ch_out, 3, 2)
            stack = tf.concat([skip_stack, stack], CHANNEL_DIM, name='concat_skip')
        return stack

    def _transition_down(self, stack, bi):
        with tf.variable_scope('TD_{}'.format(bi)) as scope:
            stack = BNReLU('bnrelu', stack)
            stack = MaxPooling('pool', stack, 2, padding='SAME')
        return stack

    def _dense_block(self, stack, n_units, init_uidx):
        unit_idx = init_uidx
        l_layers = []
        for ui in range(n_units):
            unit_idx += 1
            scope_name = self.compute_scope_basename(unit_idx)
            with tf.variable_scope(scope_name+'.feat'):
                stack = BNReLU('bnrelu', stack)
                l = Conv2D('conv3x3', stack, self.growth_rate, 3)
                l = Dropout('dropout', l, keep_prob=0.8)
                stack = tf.concat([stack, l], CHANNEL_DIM, name='concat_feat')
                l_layers.append(l)
        return stack, unit_idx, l_layers
    
    def _compute_ll_feats(self, image):
        # compute init feature
        l_feats = self._compute_init_l_feats(image)
        stack = l_feats[0]

        n_units_per_block = self.network_config.n_units_per_block

        # compute downsample blocks
        unit_idx = -1
        skip_connects = []
        for bi in range(self.n_pools):
            n_units = n_units_per_block[bi]
            stack, unit_idx, l_layers = self._dense_block(stack, n_units, unit_idx)
            skip_connects.append(stack)
            stack = self._transition_down(stack, bi)

        # center block
        skip_connects = list(reversed(skip_connects))
        n_units = n_units_per_block[self.n_pools]
        stack, unit_idx, l_layers = self._dense_block(stack, n_units, unit_idx) 

        # upsampling blocks
        for bi in range(self.n_pools):
            stack = self._transition_up(skip_connects[bi], l_layers, bi)
            n_units = n_units_per_block[bi+self.n_pools+1] 
            stack, unit_idx, l_layers = self._dense_block(stack, n_units, unit_idx)

        # interface with AnytimeFCN 
        # creat dummy feature for previous layers, and use stack as final feat
        ll_feats = [ [None] for i in range(unit_idx+1) ]
        stack = BNReLU('bnrelu_final', stack)
        ll_feats[-1] = [stack]
        return ll_feats

    def _get_optimizer(self):
        assert self.options.init_lr > 0, self.options.init_lr
        lr = get_scalar_var('learning_rate', self.options.init_lr, summary=True)
        opt = tf.train.RMSPropOptimizer(lr, 0.995)
        return opt


class AnytimeFCDensenet(AnytimeFCN, AnytimeDensenet):
    """
        Anytime FC-densenet. Use AnytimeFCN to have FC input, logits, costs. 

        Use AnytimeDensenet to have dense_select_indices, compute_transition, 
        compute_block

        Implement compute_ll_feats here to organize the feats
    """

    def __init__(self, args):
        # set up params from regular densenet.
        AnytimeDensenet.__init__(self, None, args)

        # Class weight for fully convolutional networks
        self.class_weight = None
        if hasattr(args, 'class_weight'):
            self.class_weight = args.class_weight
        if self.class_weight is None:
            self.class_weight = np.ones(self.num_classes, dtype=np.float32) 
        logger.info('Class weights: {}'.format(self.class_weight))
        
        # FC-dense specific
        self.n_pools = args.n_pools
        
        # other format is not supported yet
        assert self.n_pools * 2 + 1 == self.n_blocks

        # FC-dense doesn't support width > 1 yet
        assert self.width == 1
        # FC-dense doesn't like the starting pooling of imagenet initial conv/pool
        assert self.network_config.s_type == 'basic'


    def compute_transition_up(self, pls, skip_pls, trans_idx):
        """ for each previous layer, transition it up with deconv2d i.e., 
            conv2d_transpose
        """
        new_pls = []
        for pli, pl in enumerate(pls):
            if pli < len(skip_pls):
                new_pls.append(skip_pls[pli])
                continue 
            # implied that skip_pls is exhausted
            with tf.variable_scope('transit_{:02d}_{:02d}'.format(trans_idx, pli)):
                ch_in = pl.get_shape().as_list()[CHANNEL_DIM]
                ch_out = int(ch_in * self.reduction_ratio)
                #kernel_shape=3, stride=2
                new_pls.append(Deconv2D('deconv', pl, ch_out, 3, 2, nl=BNReLU))
        return new_pls


    def _compute_ll_feats(self, image):
        """
            This section transcribe SimJeg's FCdensnet construction to the 
            tensorflow framework. https://github.com/SimJeg/FC-DenseNet

            It also changes how TD/TU layers work, since
            log-dense computes TD/TU for each individual layer instead of all 
            previous layers.

            BNReLUConv in the original. ConvBNReLU here. 

        """
        l_feats = self._compute_init_l_feats(image)
        pls = [l_feats[0]]
        l_pls = []
        pmls = []
        for bi, n_units in enumerate(self.network_config.n_units_per_block):  
            pls, pmls = self.compute_block(pls, pmls, n_units)
            if bi < self.n_pools: 
                # downsampling
                l_pls.append(pls)
                pls = self.compute_transition(pls, bi)
            elif bi < self.n_blocks - 1:
                # To check: first deconv at self.n_pools, every layer except the last block
                # has pl of featmap-size of n_pools-1. 
                # The second to the last block has the final upsampling, and it has 
                # bi = 2*n_pools - 1; The featmap size matches that of l_pls[0] 
                skip_pls = l_pls[2*self.n_pools - bi - 1]
                pls = self.compute_transition_up(pls, skip_pls, bi)
        
        ll_feats = [ [ feat ] for feat in pmls ]
        assert len(ll_feats) == self.total_units
        return ll_feats

    def _get_optimizer(self):
        assert self.options.init_lr > 0, self.options.init_lr
        lr = get_scalar_var('learning_rate', self.options.init_lr, summary=True)
        opt = tf.train.RMSPropOptimizer(lr, 0.995)
        return opt

