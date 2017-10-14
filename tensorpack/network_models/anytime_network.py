import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils import anytime_loss, logger, utils, fs
from tensorpack.callbacks import Exp3CPU, RWMCPU, FixedDistributionCPU, ThompsonSamplingCPU

from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import xavier_initializer
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
        elif options.depth == 26:
            n_units_per_block = [2,2,2,2]
            b_type = 'bottleneck'
        elif options.depth == 14:
            n_units_per_block = [1,1,1,1]
            b_type = 'bottleneck'
        else:
            raise ValueError('depth {} must be in [18, 34, 50, 101, 152, 26, 14]'\
                .format(options.depth))
        s_type = 'imagenet' 
        return NetworkConfig(n_units_per_block, b_type, s_type)

    elif hasattr(options, 'densenet_depth') and options.densenet_depth is not None:
        if options.densenet_depth == 121:
            n_units_per_block = [6, 12, 24, 16]
        elif options.densenet_depth == 169:
            n_units_per_block = [6, 12, 32, 32]
        elif options.densenet_depth == 201:
            n_units_per_block = [6, 12, 48, 32]
        elif options.densenet_depth == 265:
            n_units_per_block = [6, 12, 64, 48]
        elif options.densenet_depth == 197:
            n_units_per_block = [16, 16, 32, 32]
        elif options.densenet_depth == 217:
            n_units_per_block = [6, 12, 56, 32]
        elif options.densenet_depth == 229:
            n_units_per_block = [16, 16, 48, 32]
        elif options.densenet_depth == 369:
            n_units_per_block = [8, 16, 80, 80]
        elif options.densenet_depth == 409:
            n_units_per_block = [6, 12, 120, 64] 
        elif options.densenet_depth == 205:
            n_units_per_block = [6, 12, 66, 16]
        else:
            raise ValueError('densenet depth {} is undefined'\
                .format(options.densenet_depth))
        b_type = 'bottleneck'
        s_type = 'imagenet'
        return NetworkConfig(n_units_per_block, b_type, s_type)#, default_growth_rate)

    elif hasattr(options, 'msdensenet_depth') and options.msdensenet_depth is not None:
        if options.msdensenet_depth == 24:
            n_units_per_block = [7, 8, 8]
            s_type = 'basic'
            # g=24
        elif options.msdensenet_depth == 23: 
            n_units_per_block = [6, 6, 5, 5]
            s_type = 'imagenet'
            # g=64
        else:
            raise ValueError('Undefined msdensenet_depth')
        b_type = 'bottleneck'
        return NetworkConfig(n_units_per_block, b_type, s_type)
 
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
        return NetworkConfig([options.num_units]*options.n_blocks, 
                             options.b_type, options.s_type)

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
    parser.add_argument('--weights_at_block_ends', 
                        help='Whether only have weights>0 at block ends, useful for fcn',
                        default=False, action='store_true') 
    parser.add_argument('--s_type', help='starting conv type',
                        type=str, default='basic', choices=['basic', 'imagenet'])
    parser.add_argument('--b_type', help='block type',
                        type=str, default='basic', choices=['basic', 'bottleneck'])
    parser.add_argument('--prediction_feature', 
                        help='Type of feature processing for prediction',
                        type=str, default='none', choices=['none', '1x1', 'msdense', 'bn'])
    parser.add_argument('--prediction_feature_ch_out_rate',
                        help='ch_out= int( <rate> * ch_in)',
                        type=np.float32, default=1.0)

    ## alternative_training_target, distillation/compression
    parser.add_argument('--alter_label', help="Type of alternative target to use",
                        default=False, action='store_true')
    parser.add_argument('--alter_loss_w', help="percentage of alter loss weight",
                        type=np.float32, default=0.5)
    parser.add_argument('--alter_label_activate_frac', 
                        help="Fraction of anytime predictions that uses alter_label",
                        type=np.float32, default=0.75)
    parser.add_argument('--high_temperature', help='Temperature for training distill targets',
                        type=np.float32, default=1.0)

    ## stop gradient / forward thinking / boost-net / no-grad
    parser.add_argument('--stop_gradient', help='Whether to stop gradients.',
                        default=False, action='store_true')
    parser.add_argument('--sg_gamma', help='Gamma for partial stop_gradient',
                        type=np.float32, default=0)

    ## selecting loss (aka ls_method, samloss) 
    parser.add_argument('--init_select_idx', help='the loss anytime_idx to select initially',
                        type=int)
    parser.add_argument('--samloss', 
                        help='Method to Sample losses to update',
                        type=int, default=6)
    parser.add_argument('--exp_gamma', help='Gamma for exp3 in sample loss',
                        type=np.float32, default=0.3)
    parser.add_argument('--sum_rand_ratio', help='frac{Sum weight}{randomly selected weight}',
                        type=np.float32, default=2.0)
    parser.add_argument('--last_reward_rate', help='rate of last reward in comparison to the max',
                        type=np.float32, default=0.85)

    ## loss_weight computation
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=FUNC_TYPE_ANN)
    parser.add_argument('--exponential_base', help='Exponential base',
                        type=np.float32)
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=-1)
    parser.add_argument('--last_weight_to_early_sum',
                        help='Final prediction  weight divided by sum of early weights',
                        type=np.float32, default=1.0)
    parser.add_argument('--normalize_weights',
                        help='method to normalize the weights.'\
                        +' last: last one will have 1. all : sum to 1. log : sum to log(N)'\
                        +' Last seems to work the best. log for back-compatibility for f=5,9,10',
                        type=str, default='last', choices=['last', 'all', 'log'])

    ## misc: training params, data-set params, speed/memory params
    parser.add_argument('--init_lr', help='The initial learning rate',
                        type=np.float32, default=0.01)
    parser.add_argument('--batch_norm_decay', help='decay rate of batchnorms',
                        type=np.float32, default=0.9)
    parser.add_argument('--num_classes', help='Number of classes', 
                        type=int, default=10)
    parser.add_argument('--regularize_coef', help='How coefficient of regularization decay',
                        type=str, default='const', choices=['const', 'decay']) 
    parser.add_argument('--regularize_const', help='Regularization constant',
                        type=float, default=1e-4)
    parser.add_argument('--w_init', help='method used for initializing W',
                        type=str, default='var_scale', choices=['var_scale', 'xavier'])
    ## Special options to force input as uint8 and do mean/std process in graph in order to save memory
    # during cpu - gpu communication
    parser.add_argument('--input_type', help='Type for input, uint8 for certain dataset to speed up',
                        type=str, default='float32', choices=['float32', 'uint8'])
    parser.add_argument('--do_mean_std_gpu_process', 
                        help='Whether use args.mean args.std to process in graph',
                        default=False, action='store_true')
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
        if self.network_config.s_type == 'imagenet' and self.init_channel < 64:
            logger.warn('Resnet imagenet requires 64 initial channels')

        self.n_blocks = len(self.network_config.n_units_per_block) 
        self.cumsum_blocks = np.cumsum(self.network_config.n_units_per_block)
        self.width = args.width
        self.num_classes = self.options.num_classes
        self.alter_label = self.options.alter_label
        self.alter_label_activate_frac = self.options.alter_label_activate_frac
        self.alter_loss_w = self.options.alter_loss_w

        self.weights = anytime_loss.loss_weights(self.total_units, args, 
            cfg=self.network_config.n_units_per_block)
        self.weights_sum = np.sum(self.weights)
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

        if self.options.w_init == 'xavier':
            self.w_init = xavier_initializer()
        elif self.options.w_init == 'var_scale':
            self.w_init = variance_scaling_initializer(mode='FAN_AVG')

        self.input_type = tf.float32 if self.options.input_type == 'float32' else tf.uint8
        if self.options.do_mean_std_gpu_process:
            if not hasattr(self.options, 'mean'):
                raise Exception('gpu_graph expects mean but it is not in the options')
            if not hasattr(self.options, 'std'):
                raise Exception('gpu_graph expects std, but it is not in the options')
    
    def _get_inputs(self):
        additional_input = []
        if self.is_model_training() and self.alter_label:
            additional_input = [InputDesc(tf.float32, [None, self.num_classes], 'alter_label')]
        return [InputDesc(self.input_type, 
                    [None, self.input_size, self.input_size, 3],'input'),
                InputDesc(tf.int32, [None], 'label')] + additional_input

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
        logger.info("AANN samples with method {}".format(self.options.ls_method))
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
        raise Exception("Invoked the base AnytimeNetwork. Use a specific one instead")

    def _compute_prediction_and_loss(self, l, label_obj, unit_idx):
        """
            l: feat_map of DATA_FORMAT 
            label_obj: target to determine the loss
            unit_idx : the feature computation unit index.
        """
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, self.num_classes, nl=tf.identity)
        if self.options.high_temperature > 1.0:
            logits /= self.options.high_temperature
            
        ## local cost/error_rate
        label = label_obj[0]
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        add_moving_summary(cost)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
        
        wrong5 = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong5, name='train-error-top5'))
        
        if self.alter_label and self.is_model_training() and \
                unit_idx < self.alter_label_activate_frac * self.total_units:
            alabel = label_obj[1]
            sq_loss = np.float32(self.num_classes) * \
                tf.losses.mean_squared_error(labels=alabel, predictions=logits)
            add_moving_summary(sq_loss, name='alter_sq_loss')
            if self.alter_loss_w != 0.0:
                 cost = cost * (1 - self.alter_loss_w) + sq_loss * self.alter_loss_w

        return logits, cost


    def _parse_inputs(self, inputs):
        """
            Parse the inputs so that it's split into image, followed by a "label" object.
            Note that the label object may contain multiple labels, such as coarse labels,
            and soft labels. 

            The first returned variable is always the image as it was inputted
        """
        image = inputs[0]
        label = inputs[1:]
        return image, label

    def _build_graph(self, inputs):
        logger.info("sampling loss with method {}".format(self.options.ls_method))
        select_idx = self._get_select_idx()
        
        with argscope([Conv2D, Deconv2D, AvgPooling, MaxPooling, BatchNorm, GlobalAvgPooling], 
                      data_format=DATA_FORMAT), \
            argscope([Conv2D, Deconv2D], nl=tf.identity, use_bias=False), \
            argscope([Conv2D], W_init=self.w_init), \
            argscope([BatchNorm], decay=self.options.batch_norm_decay):

            image, label = self._parse_inputs(inputs)

            # Common GPU side preprocess (uint8 -> float32), mean std, NCHW.
            if self.input_type == tf.uint8:
                image = tf.cast(image, tf.float32) * (1.0 / 255)
            if self.options.do_mean_std_gpu_process:
                if self.options.mean is not None:
                    image = image - tf.constant(self.options.mean, dtype=tf.float32) 
                if self.options.std is not None:
                    image = image / tf.constant(self.options.std, dtype=tf.float32)
            if DATA_FORMAT == 'NCHW':
                image = tf.transpose(image, [0,3,1,2])

            self.dynamic_batch_size = tf.identity(tf.shape(image)[0], name='dynamic_batch_size')
            ll_feats = self._compute_ll_feats(image)
            
            if self.options.stop_gradient:
                # NOTE:
                # Do not regularize for stop-gradient case, because
                # stop-grad requires cycling lr, and switching training targets
                wd_w = 0
            elif self.options.regularize_coef == 'const':
                wd_w = self.options.regularize_const
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
                        ch_in = l.get_shape().as_list()[CHANNEL_DIM]
                        if self.options.prediction_feature == '1x1':
                            ch_out = int(self.options.prediction_feature_ch_out_rate * ch_in)
                            l = Conv2D('conv1x1', l, ch_out, 1)
                            l = BNReLU('bnrelu1x1', l)
                        elif self.options.prediction_feature == 'msdense':
                            if self.network_config.s_type == 'basic':
                                ch_inter = 128
                            else:
                                ch_inter = ch_in
                            l = Conv2D('conv1x1_0', l, ch_inter, 3, stride=2)
                            l = BNReLU('bnrelu1x1_0', l)
                            l = Conv2D('conv1x1_1', l, ch_inter, 3, stride=2)
                            l = BNReLU('bnrelu1x1_1', l)
                        elif self.options.prediction_feature == 'bn':
                            l = BatchNorm('bn', l)
                        
                        logits, cost = self._compute_prediction_and_loss(l, label, unit_idx)
                    #end scope of layer.w.pred

                    ## Compute the contribution of the cost to total cost
                    # Additional weight for unit_idx. 
                    add_weight = 0
                    if select_idx is not None:
                        add_weight = tf.cond(tf.equal(anytime_idx, 
                                                      select_idx),
                            lambda: tf.constant(self.weights_sum, 
                                                dtype=tf.float32),
                            lambda: tf.constant(0, dtype=tf.float32))
                    if self.options.sum_rand_ratio > 0:
                        total_cost += (cost_weight + add_weight / \
                            self.options.sum_rand_ratio) * cost
                    else:
                        total_cost += add_weight * cost

                    ## Regularize weights from FC layers.
                    wd_cost += cost_weight * wd_w * tf.nn.l2_loss(logits.variables.W)
                    wd_cost += cost_weight * wd_w * tf.nn.l2_loss(logits.variables.b)

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
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss))
        wd_cost = tf.identity(wd_cost, name='wd_cost')
        total_cost = tf.identity(total_cost, name='sum_losses')
        add_moving_summary(total_cost, wd_cost)
        self.cost = tf.add_n([total_cost, wd_cost], name='cost') # specify training loss
        # monitor W # Too expensive in disk space :-/
        #add_param_summary(('.*/W', ['histogram']))   


    def _get_optimizer(self):
        assert self.options.init_lr > 0, self.options.init_lr
        lr = get_scalar_var('learning_rate', self.options.init_lr, summary=True)
        opt = None
        if hasattr(self.options, 'optimizer'):
            if self.options.optimizer == 'rmsprop':
                logger.info('RMSPropOptimizer')
                opt = tf.train.RMSPropOptimizer(lr)
        if opt is None:
            logger.info('No optimizer was specified, using default MomentumOptimizer')
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
                #add_moving_summary(weight_i)
        return select_idx


class AnytimeResnet(AnytimeNetwork):
    def __init__(self, input_size, args):
        super(AnytimeResnet, self).__init__(input_size, args)

    def residual_basic(self, name, l_feats, increase_dim=False):
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
                # The first round doesn't use pre relu according to pyramidial deep net
                l = tf.nn.relu(l)
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
                # according to pyramidal net, do not use relu here
                l = tf.nn.relu(l)
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
                    l_feats = self.residual_basic(scope_name, l_feats, \
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
    parser.add_argument('--densenet_version', help='specify the version of densenet to use',
                        type=str, default='atv1', choices=['atv1', 'atv2', 'dense', 'loglog'])
    parser.add_argument('-g', '--growth_rate', help='growth rate k for log dense',
                        type=int, default=16)
    parser.add_argument('--bottleneck_width', help='multiplier of growth for width of bottleneck',
                        type=float, default=4.0)
    parser.add_argument('--growth_rate_multiplier', 
                        help='a constant to multiply growth_rate by at pooling',
                        type=int, default=1, choices=[1,2])
    parser.add_argument('--use_init_ch', 
                        help='whether to use specified init channel argument, '\
                            +' useful for networks that has specific init_ch based on'\
                            +' other metrics such as densenet',
                        default=False, action='store_true')
    parser.add_argument('--dense_select_method', help='densenet previous feature selection choice',
                        type=int, default=0)
    parser.add_argument('--early_connect_type', help='Type of forced early conneciton_types',
                        type=int, default=0)
    parser.add_argument('--log_dense_coef', help='The constant multiplier of log(depth) to connect',
                        type=np.float32, default=1)
    parser.add_argument('--log_dense_base', help='base of log',
                        type=np.float32, default=2)
    parser.add_argument('--reduction_ratio', help='reduction ratio at transitions',
                        type=np.float32, default=1)
    parser.add_argument('--transition_batch_size', 
                        help='number of layers to transit together per conv; ' +\
                             '-1 means all previous layers transition together using 1x1 conv',
                        type=int, default=1)
    parser.add_argument('--use_transition_mask',
                        help='When transition together, whether use W_mask to force indepence',
                        default=False, action='store_true')
    parser.add_argument('pre_activate', help='whether BNReLU pre conv or after',
                        default=False, action='store_true')
    return parser, depth_group


class AnytimeDensenet(AnytimeNetwork):
    def __init__(self, input_size, args):
        super(AnytimeDensenet, self).__init__(input_size, args)
        self.dense_select_method = self.options.dense_select_method
        self.log_dense_coef = self.options.log_dense_coef
        self.log_dense_base = self.options.log_dense_base
        self.reduction_ratio = self.options.reduction_ratio
        self.growth_rate = self.options.growth_rate
        self.growth_rate_multiplier = self.options.growth_rate_multiplier
        self.bottleneck_width = self.options.bottleneck_width
        self.early_connect_type = self.options.early_connect_type

        ## deprecated. don't use
        self.transition_batch_size = self.options.transition_batch_size
        self.use_transition_mask = self.options.use_transition_mask

        self.pre_activate = self.options.pre_activate

        if not self.options.use_init_ch:
            default_ch = self.growth_rate * 2
            if self.init_channel != default_ch:
                self.init_channel = default_ch
                logger.info("Densenet sets the init_channel to be " \
                    + "2*growth_rate by default. " \
                    + "I'm setting this automatically!")
        
        # width > 1 is not implemented for densenet
        assert self.width == 1,self.width

        if self.options.func_type == FUNC_TYPE_ANN \
            and self.options.ls_method != BEST_AANN_METHOD:
            logger.warn("Densenet prefers using AANN instead of other methods."\
                +"Changing samloss to be {}".format(BEST_AANN_METHOD))
            self.options.ls_method = BEST_AANN_METHOD
            self.options.samloss = BEST_AANN_METHOD

        ## pre-compute connection 
        self._connections = None
        self.connections = None # For each ui, an input list
        self.l_max_scale = None # exists for each pls, including the init conv
        self.pre_compute_connections()


    ## whether pls[x] should be included for computing ui,
    # given that pls[x] is selected already by DSM, early forcement.
    # e.g., log-dense-v2 uses this to cut early connections to do block
    # compression
    def special_filter(self, ui, x):
        return True


    ## Some connections methods requrie some recursion or complex 
    # functions to set all the connection together first.
    # this will be set in self._connections.
    # it will then be used by pre_compute_connections and 
    # dense_select_method to be augmented with forced connections
    # and special_filter
    def _pre_compute_connections(self):
        self._connections = None


    def pre_compute_connections(self):
        self._pre_compute_connections()
        self.connections = []
        self.l_max_scale = [ 0 for _ in range(self.total_units + 1) ]
        curr_bi = 0
        for ui in range(self.total_units):
            if self.cumsum_blocks[curr_bi] == ui:
                curr_bi += 1
            self.connections.append(self.dense_select_indices(ui))
            for i in self.connections[-1]:
                self.l_max_scale[i] = max(self.l_max_scale[i], curr_bi)


    def dense_select_indices(self, ui):
        if self._connections is not None:
            # the selections are precomputed. 
            # get them for the stored list
            indices = self._connections[ui]
        else:
            # the selection is not defined yet. 
            # compute dynamically right now
            indices = self._dense_select_indices(ui)
        indices = self._dense_select_early_connect(ui, indices)
        indices = filter(lambda x, ui=ui: x <=ui and x >=0 and \
            self.special_filter(ui, x), np.unique(indices))
        return indices


    def _dense_select_early_connect(self, ui, indices):
        if self.early_connect_type == 0:
            # default 0 does nothing
            pass
        if self.early_connect_type % 2 ==  1:  #e.g., 1
            # force connection to end of first block
            indices.append(self.network_config.n_units_per_block[0])
        if (self.early_connect_type >> 1) % 2 == 1:  #e.g., 2
            # force connect to all of first block
            indices.extend(list(range(self.network_config.n_units_per_block[0]+1))) 
        if (self.early_connect_type >> 2) % 2 == 1:  #e.g., 4
            # force connect to end of the first three blocks 
            indices.extend(self.cumsum_blocks[:3])
        if (self.early_connect_type >> 3) % 2 == 1: # e.g., 8
            # force connect to end of all blocks 
            indices.extend(self.cumsum_blocks)

        indices = filter(lambda x : x <=ui and x >=0, np.unique(indices))
        return indices


    def _dense_select_indices(self, ui):
        """
            Given ui, return the list of indices i's such that 
            pls[i] contribute to forming layer[ui] i.e. pls[ui+1].

            For methods that can be computed directly using ui, and other
            args, use this method to do so. 

            If the computation is too complex or it is easy to pre-compute
            all connections before hand, see loglogdense as an example, and
            use _pre_compute_connections.

            ui : unit_idx
        """
        if self.dense_select_method == 0:
            # log dense
            if not hasattr(self, 'exponential_diffs'):
                df = 1
                exponential_diffs = [0]
                while True:
                    df = df * self.log_dense_base
                    if df > self.total_units * self.log_dense_coef:
                        break
                    int_df = int((df-1) / self.log_dense_coef)
                    if int_df != exponential_diffs[-1]:
                        exponential_diffs.append(int_df)
                self.exponential_diffs = exponential_diffs
            diffs = self.exponential_diffs 
        elif self.dense_select_method == 1:
            # all at the end with log(i)
            diffs = list(range(int(np.log2(ui + 1) \
                / np.log2(self.log_dense_base) * self.log_dense_coef) + 1))
        elif self.dense_select_method == 2:
            # all at the end with log(L)
            n_select = int((np.log2(self.total_units +1) \
                / np.log2(self.log_dense_base) + 1) * self.log_dense_coef)
            diffs = list(range(int(np.log2(self.total_units + 1)) + 1))
        elif self.dense_select_method == 3:
            # Evenly spaced connections
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
            # select all
            diffs = list(range(ui+1))
        elif self.dense_select_method == 5: 
            # mini dense (only close to the last layer) 
            diffs = [0, 1]
            left = 0
            right = self.total_units + 1
            while right != ui+2: # seek ui+1
                mid = (left + right) // 2
                if ui+1 >= mid:
                    left = mid
                else:
                    right = mid
            df = right - (right + left) // 2 - 1
            if df > 1:
                diffs.append(df)

        indices = [ui - df  for df in diffs if ui - df >= 0 ]
        return indices


    def compute_block(self, pls, pmls, n_units, growth):
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
                sl_indices = self.connections[unit_idx]
                logger.info("unit_idx = {}, len past_feats = {}, selected_feats: {}".format(\
                    unit_idx, len(pls), sl_indices))
                
                ## Question: TODO whether save the pre-bnrelu feat or after-bnrelu feat
                # The current version saves the after-bnrelue to save mem/computation
                ml = tf.concat([pls[sli] for sli in sl_indices], \
                               CHANNEL_DIM, name='concat_feat')
                # pre activation
                if self.pre_activate:
                    ml = BNReLU('bnrelu_merged', ml)
                    nl = tf.identity
                else:
                    nl = BNReLU
                if self.network_config.b_type == 'bottleneck':
                    bottleneck_width = int(self.options.bottleneck_width * growth)
                    #ch_in = ml.get_shape().as_list()[CHANNEL_DIM]
                    #bottleneck_width = min(ch_in, bottleneck_width)
                    l = (LinearWrap(ml)
                        .Conv2D('conv1x1', bottleneck_width, 1, nl=BNReLU)
                        .Dropout('dropout', keep_prob=0.8)
                        .Conv2D('conv3x3', growth, 3, nl=nl)())
                else:
                    l = Conv2D('conv3x3', ml, growth, 3, nl=nl)
                l = Dropout('dropout', l, keep_prob=0.8)
                pls.append(l)

                # If the feature is used for prediction, store it.
                if self.weights[unit_idx] > 0:
                    if self.pre_activate:
                        l = BNReLU('bnrelu_local', l)
                    pmls.append(tf.concat([ml, l], CHANNEL_DIM, name='concat_pred'))
                else:
                    pmls.append(None)
        return pls, pmls


    ##
    # pls (list) : list of previous layers including the init conv
    # trans_idx (int) : curret block index bi, (the one just finished)
    def compute_transition(self, pls, trans_idx):
        new_pls = []
        for pli, pl in enumerate(pls):
            if self.l_max_scale[pli] <= trans_idx: 
                new_pls.append(None)
                continue

            ch_in = pl.get_shape().as_list()[CHANNEL_DIM]
            ch_out = int(ch_in * self.growth_rate_multiplier * self.reduction_ratio)

            with tf.variable_scope('transit_{:02d}_{:02d}'.format(trans_idx, pli)): 
                if self.pre_activate:
                    pl = BNReLU('bnrelu_transit', pl)
                    nl = tf.identity
                else:
                    nl = BNReLU
                new_pl = (LinearWrap(pl)
                    .Conv2D('conv', ch_out, 1, nl=nl)
                    .Dropout('dropout', keep_prob=0.8)
                    .AvgPooling('pool', 2, padding='SAME')())
                new_pls.append(new_pl)
        return new_pls

            
    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        pls = [l_feats[0]]
        pmls = []
        growth = self.growth_rate
        for bi, n_units in enumerate(self.network_config.n_units_per_block):  
            pls, pmls = self.compute_block(pls, pmls, n_units, growth)
            if bi != self.n_blocks - 1: 
                growth *= self.growth_rate_multiplier
                pls = self.compute_transition(pls, bi)
        
        ll_feats = [ [ feat ] for feat in pmls ]
        assert len(ll_feats) == self.total_units
        return ll_feats



class DenseNet(AnytimeDensenet):
    """
        This class is for reproducing densenet results. 
        There is no choices of selecting connections as in AnytimeDensenet
    """
    def __init__(self, input_size, args):
        super(DenseNet, self).__init__(input_size, args)


    def compute_block(self, pmls, layer_idx, n_units, growth):
        pml = pmls[-1]
        if layer_idx > -1:
            with tf.variable_scope('transit_after_{}'.format(layer_idx)) as scope:
                ch_in = pml.get_shape().as_list()[CHANNEL_DIM]
                ch_out = int(ch_in * self.reduction_ratio)
                pml = Conv2D('conv1x1', pml, ch_out, 1, nl=BNReLU)
                pml = Dropout('dropout', pml, keep_prob=0.8)
                pml = AvgPooling('pool', pml, 2, padding='SAME')

        for k in range(n_units):
            layer_idx +=1
            scope_name = self.compute_scope_basename(layer_idx)
            with tf.variable_scope(scope_name) as scope:
                l = pml
                if self.network_config.b_type == 'bottleneck':
                    bnw = int(self.bottleneck_width * growth)
                    l = Conv2D('conv1x1', l, bnw, 1, nl=BNReLU)
                    l = Dropout('dropout', l, keep_prob=0.8)
                l = Conv2D('conv3x3', l, growth, 3, nl=BNReLU)
                l = Dropout('dropout', l, keep_prob=0.8)
                pml = tf.concat([pml, l], CHANNEL_DIM, name='concat')
                pmls.append(pml)
        return pmls

    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        pmls = [l_feats[0]]
        growth = self.growth_rate
        layer_idx = -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            pmls = self.compute_block(pmls, layer_idx, n_units, growth)
            layer_idx += n_units

        ll_feats = [ [ml] for ml in pmls]
        return ll_feats[1:]


class AnytimeLogDensenetV2(AnytimeDensenet):
    """
        This version of dense net will do block compression 
        by compression a block into log L layers. 
        Any future layer will use the entire log L layers,
        even in log-dense
    """
    def __init__(self, input_size, args):
        super(AnytimeLogDensenetV2, self).__init__(input_size, args)

    
    def special_filter(self, ui, x):
        if ui == 0 or x == 0: 
            return False
        bi = bisect.bisect_right(self.cumsum_blocks, ui)
        bi_x = bisect.bisect_right(self.cumsum_blocks, x-1)
        return bi == bi_x

    def compute_block(self, layer_idx, n_units, l_mls, bcml, growth):
        pls = []
        # offset is the first layer_idx that in this block. 
        # layer_idx+1 now contains the last of the last block
        pli_offset = layer_idx + 2
        for k in range(n_units):
            layer_idx += 1
            scope_name = self.compute_scope_basename(layer_idx)
            with tf.variable_scope(scope_name):
                sl_indices = self.connections[layer_idx]
                logger.info("layer_idx = {}, len past_feats = {}, selected_feats: {}".format(\
                    layer_idx, len(pls), sl_indices))

                ml = tf.concat([bcml] + [pls[sli - pli_offset] for sli in sl_indices], \
                               CHANNEL_DIM, name='concat_feat')
                if self.pre_activate:
                    ml = BNReLU('bnrelu_merged', ml) 
                    nl = tf.identity
                else:
                    nl = BNReLU
                if self.network_config.b_type == 'bottleneck':
                    bnw = int(self.bottleneck_width * growth)
                    l = Conv2D('conv1x1', ml, bnw, 1, nl=BNReLU)
                    l = Dropout('dropout', l, keep_prob=0.8)
                    l = Conv2D('conv3x3', l, growth, 3, nl=nl)
                else:
                    l = Conv2D('conv3x3', ml, growth, 3, nl=nl) 
                # dense connections need drop out to regularize
                l = Dropout('dropout', l, keep_prob=0.8)
                pls.append(l)

                if self.weights[layer_idx] > 0:
                    if self.pre_activate:
                        l = BNReLU('bnrelu_local', l)
                    ml = tf.concat([ml, l], CHANNEL_DIM, name='concat_pred')
                    l_mls.append(ml)
                else:
                    l_mls.append(None) 
        return pls, l_mls


    def update_compressed_feature(self, layer_idx, ch_out, pls, bcml):
        """
            pls: new layers
            bcml : the compressed features for generating pls
        """
        with tf.variable_scope('transition_after_{}'.format(layer_idx)) as scope: 
            l = tf.concat(pls, CHANNEL_DIM, name='concat_new')
            ch_new = l.get_shape().as_list()[CHANNEL_DIM]
            if self.pre_activate:
                l = BNReLU('pre_bnrelu', l)
                bcml = BNReLU('pre_bnrelu_old', bcml)
                nl = tf.identity
            else:
                nl = BNReLU
            l = Conv2D('conv1x1_new', l, min(ch_out, ch_new), 1, nl=nl)
            l = Dropout('dropout_new', l, keep_prob=0.8)
            l = AvgPooling('pool_new', l, 2, padding='SAME')

            ch_old = bcml.get_shape().as_list()[CHANNEL_DIM]
            bcml = Conv2D('conv1x1_old', bcml, ch_old, 1, nl=nl)
            bcml = Dropout('dropout_old', bcml, keep_prob=0.8)
            bcml = AvgPooling('pool_old', bcml, 2, padding='SAME') 
            
            bcml = tf.concat([bcml, l], CHANNEL_DIM, name='concat_all')
        return bcml


    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        bcml = l_feats[0]
        l_mls = []
        growth = self.growth_rate
        layer_idx = -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):  
            pls, l_mls = self.compute_block(layer_idx, n_units, l_mls, bcml, growth)
            layer_idx += n_units
            if bi != self.n_blocks - 1: 
                ch_out = growth * (int(np.log2(self.total_units + 1)) + 1)
                bcml = self.update_compressed_feature(layer_idx, ch_out, pls, bcml)
        
        ll_feats = [ [ml] for ml in l_mls ]
        assert len(ll_feats) == self.total_units
        return ll_feats

############################
# The craziness of Log-Log dense
############################
class AnytimeLogLogDenseNet(AnytimeDensenet):
    def __init__(self, input_size, args):
        super(AnytimeLogLogDenseNet, self).__init__(input_size, args)

    ## pre compute connections for log-log dense as it is rather complicated
    # construct connection adjacency list for backprop
    # The pls starts with the initial conv which has index 0, so that it is 1-based. 
    # The input ui is 0-based. Hence, e.g., ui (input) always connects to 
    # pls[ui], which is the previous layer of input[ui]. 
    def _pre_compute_connections(self):
        # Everything is connected to the previous layer
        # l_adj[0] is a placeholder, so ignore that it connects to -1.
        l_adj = [ [i-1] for i in range(self.total_units+1) ]

        ## padding connection offset to ensure at least bn_width connections are used.
        # i - offset is the input index
        padding_offsets = [1, 2, 4, 8, 16, 32]

        
        ## update l_adj connecitono on interval [a, b)
        def loglog_connect(a, b, force_connect_locs=[]):
            if b-a <= 2:
                return None
                
            seg_len = b-a
            step_len = int(np.sqrt(b-a))
            key_indices = list(range(a, b, step_len))
            if len(force_connect_locs) > 0:
                for fc_key in force_connect_locs:
                    if not fc_key in key_indices:
                        key_indices.append(fc_key)
                key_indices = sorted(key_indices)
            if key_indices[-1] != b-1:
                key_indices.append(b-1)

            # connection at the current recursion depth
            for ki, key in enumerate(key_indices):
                if ki == 0:
                     continue
                for prev_key in key_indices[:ki]:
                    if not prev_key in l_adj[key]:
                        l_adj[key].append(prev_key)
                prev_key = key_indices[ki-1]
                for li in range(prev_key + 1, key):
                    if not prev_key in l_adj[li]:
                        l_adj[li].append(prev_key)
                loglog_connect(prev_key, key+1)
            return None
        
        
        force_connect_locs = self.cumsum_blocks
        loglog_connect(0, self.total_units+1, force_connect_locs)
        ## since the first conv is init conv that we don't count for pred.
        for i in range(self.total_units):
            l_adj[i+1] = filter(lambda x: x >= 0 and x <= i, np.unique(l_adj[i+1]))
            if len(l_adj[i+1]) < self.bottleneck_width:
                for offset in padding_offsets:
                    idx = i + 1 - offset
                    if idx < 0:
                        break
                    if not idx in l_adj[i+1]:
                        l_adj[i+1].append(idx)
                        if len(l_adj[i+1]) >= self.bottleneck_width:
                            break
                
        self._connections = l_adj[1:]
        
    

################################################
# FCN for semantic labeling
#
# NOTE:
#   Since the layer H/W are induced based on 
#   the input layer rn, DO NOT give inputs of 
#   H/W that are not divisible by 2**n_pools.
#   Otherwise the upsampling results in H/W that
#   mismatch the input H/W. (And screw dynamic
#   shape checking in tensorflow....)
################################################
def parser_add_fcn_arguments(parser):
    parser.add_argument('--is_label_one_hot', 
        help='whether label img contains distribution of labels',
        default=False, action='store_true')
    parser.add_argument('--eval_threshold', 
        help='The minimum valid labels weighting in [0,1] to trigger evaluation',
        type=np.float32, default=0.5) 
    parser.add_argument('--n_pools',
        help='number of pooling blocks on the cfg.n_units_per_block',
        type=int, default=None)
    return parser

class AnytimeFCN(AnytimeNetwork):
    """
        Overload AnytimeNetwork from classification set-up to semantic labeling.
    """
    def __init__(self, args):
        super(AnytimeFCN, self).__init__(None, args)

        # Class weight for fully convolutional networks
        self.class_weight = None
        if hasattr(args, 'class_weight'):
            self.class_weight = args.class_weight
        if self.class_weight is None:
            self.class_weight = np.ones(self.num_classes, dtype=np.float32) 
        logger.info('Class weights: {}'.format(self.class_weight))

        self.is_label_one_hot = args.is_label_one_hot
        self.eval_threshold = args.eval_threshold

        # TODO
        # args should contain the size of the image, so that deconv can decide 
        # whether to crop based on the sizes.
        # Do not use these shape to set the input size though, because this
        # would limit the input size, and make the batch-norm info void for
        # different size of inputs
        
        # NOTE
        # label_img is always NHWC or NHW
        # If label_img is NHWC, the distribution doesn't include void. 
        # Furthermore, label_img is 0-vec for void labels
                
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
                                scope_name+'prob_sqr_err:0',
                                scope_name+'cross_entropy_loss:0'],
                            weight_name='dynamic_batch_size:0',
                            prefix='val_'))
        return vcs
        

    def _get_inputs(self):
        if self.options.is_label_one_hot:
            # the label one-hot is in fact a distribution of labels. 
            # Void labeled pixels have 0-vector distribution.
            label_desc = InputDesc(tf.float32, 
                [None, None, None, self.num_classes], 'label')
        else:
            label_desc = InputDesc(tf.int32, [None, None, None], 'label')
        return [InputDesc(self.input_type, [None, None, None, 3], 'input'), label_desc]


    def _parse_inputs(self, inputs):
        image, label_img = inputs
        if not self.options.is_label_one_hot: 
            # From now on label_img is tf.float one hot, void has 0-vector.
            # because we assume void >=num_classes
            # Note axis=-1 b/c label is NHWC always
            label_img = tf.one_hot(label_img, self.num_classes, axis=-1)

        def nonvoid_mask(prob_img, name=None):
            # note axis=-1 b/c label img is always NHWC
            mask = tf.cast(tf.greater(tf.reduce_sum(prob_img, axis=-1), 
                                      self.options.eval_threshold), 
                           dtype=tf.float32)
            mask = tf.reshape(mask, [-1], name=name)
            # TODO is this actually beneficial; and which KeepProb to use?
            #mask = Dropout(name, mask, keep_prob=0.5)
            return mask

        def flatten_label(prob_img, name=None):
            return tf.reshape(prob_img, [-1, self.num_classes], name=name)

        l_mask = []
        l_label = []
        l_dyn_hw = []
        label_img = tf.identity(label_img, name='label_img_0')

        for pi in range(self.n_pools+1):
            l_mask.append(nonvoid_mask(label_img, 'eval_mask_{}'.format(pi)))
            l_label.append(flatten_label(label_img, 'label_{}'.format(pi)))
            img_shape = tf.shape(label_img)
            # Note that input is always NHWC. 
            l_dyn_hw.append([img_shape[1], img_shape[2]])
            if pi == self.n_pools:
                break
            label_img = AvgPooling('label_img_{}'.format(pi+1), label_img, 2, \
                                   padding='SAME', data_format='NHWC')

        return image, [l_label, l_mask, l_dyn_hw]


    def _compute_prediction_and_loss(self, l, label_inputs, unit_idx):
        l_label, l_eval_mask, l_dyn_hw = label_inputs
        # Assume all previous layers have gone through BNReLU, so conv directly
        l = Conv2D('linear', l, self.num_classes, 1, use_bias=True)
        logit_vars = l.variables
        if DATA_FORMAT == 'NCHW':
            l = tf.transpose(l, [0,2,3,1]) 
        logits = tf.reshape(l, [-1, self.num_classes], name='logits')
        logits.variables = logit_vars

        # compute  block idx
        layer_idx = unit_idx // self.width
        # first idx that is > layer_idx
        bi = bisect.bisect_right(self.cumsum_blocks, layer_idx)

        # Case downsample check: n_pool uses label_idx=n_pool
        #   0 uses 0
        # Case upsample check: bi == n_pools uses label_idx=n_pools. 
        #   the final bi == n_pools * 2 uses 0
        label_img_idx = bi
        if bi > self.n_pools:
            label_img_idx = 2 * self.n_pools - bi

        label = l_label[label_img_idx] # note this is a probability of label distri
        eval_mask = l_eval_mask[label_img_idx]
        dyn_hw = l_dyn_hw[label_img_idx]

        n_non_void_samples = tf.reduce_sum(eval_mask)
        n_non_void_samples += tf.cast(tf.less_equal(n_non_void_samples, 1e-12), tf.float32)
            
        ## Local cost/loss for training

        # Square error between distributions. 
        # Implement our own here b/c class weighting.
        prob = tf.nn.softmax(logits, name='pred_prob')
        prob_img_shape = tf.stack([-1,  dyn_hw[0], dyn_hw[1], self.num_classes])
        prob_img = tf.reshape(prob, prob_img_shape, name='pred_prob_img') 
        sqr_err = tf.reduce_sum(\
            tf.multiply(tf.square(label - prob), self.class_weight), \
            axis=1, name='pixel_prob_square_err')
        sqr_err = tf.divide(tf.reduce_sum(sqr_err * eval_mask), n_non_void_samples,
            name='prob_sqr_err')
        add_moving_summary(sqr_err)

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
        cross_entropy = cross_entropy * eval_mask
        cross_entropy = tf.divide(tf.reduce_sum(cross_entropy), n_non_void_samples,
                                  name='cross_entropy_loss')
        add_moving_summary(cross_entropy)

        # Unweighted total abs diff
        sum_abs_diff = sum_absolute_difference(prob, label)
        sum_abs_diff *= eval_mask 
        sum_abs_diff = tf.divide(tf.reduce_sum(sum_abs_diff), 
                                 n_non_void_samples, name='sum_abs_diff')
        add_moving_summary(sum_abs_diff)
        
        # confusion matrix for iou and pixel level accuracy
        int_pred = tf.argmax(logits, 1, name='int_pred')
        int_label = tf.argmax(label, 1, name='int_label')
        cm = tf.confusion_matrix(labels=int_label, predictions=int_pred,\
            num_classes=self.num_classes, name='confusion_matrix', weights=eval_mask)

        # pixel level accuracy
        accu = tf.divide(tf.cast(tf.reduce_sum(tf.diag_part(cm)), dtype=tf.float32), \
                         n_non_void_samples, name='accuracy')
        add_moving_summary(accu)

        return logits, cross_entropy


###########
## FCDense
def parser_add_fcdense_arguments(parser):
    parser = parser_add_fcn_arguments(parser)
    parser, depth_group = parser_add_densenet_arguments(parser)
    depth_group.add_argument('--fcdense_depth',
                            help='depth of the network in number of conv',
                            type=int)
    return parser, depth_group


## 
# This class reproduces tiramisu FC-Densenet for scene parsing. 
# It is meant for a comparison against anytime FCN 
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

        # TODO This version doesn't support anytime prediction (yet) 
        assert self.options.func_type == FUNC_TYPE_OPT
    
    def _transition_up(self, skip_stack, l_layers, bi):
        with tf.variable_scope('TU_{}'.format(bi)) as scope:
            stack = tf.concat(l_layers, CHANNEL_DIM, name='concat_recent')
            ch_out = stack.get_shape().as_list()[CHANNEL_DIM]
            dyn_h = tf.shape(skip_stack)[HEIGHT_DIM]
            dyn_w = tf.shape(skip_stack)[WIDTH_DIM]
            stack = Deconv2D('deconv', stack, ch_out, 3, 2, dyn_hw=[dyn_h, dyn_w])
            stack = tf.concat([skip_stack, stack], CHANNEL_DIM, name='concat_skip')
        return stack

    def _transition_down(self, stack, bi):
        with tf.variable_scope('TD_{}'.format(bi)) as scope:
            stack = BNReLU('bnrelu', stack)
            ch_in = stack.get_shape().as_list()[CHANNEL_DIM]
            stack = Conv2D('conv1x1', stack, ch_in, 1, use_bias=True)
            stack = Dropout('dropout', stack, keep_prob=0.8)
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
                l = Conv2D('conv3x3', stack, self.growth_rate, 3, use_bias=True)
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



class AnytimeFCDensenet(AnytimeFCN, AnytimeDensenet):
    """
        Anytime FC-densenet. Use AnytimeFCN to have FC input, logits, costs. 

        Use AnytimeDensenet to have dense_select_indices, compute_transition, 
        compute_block

        Implement compute_ll_feats here to organize the feats
    """

    def __init__(self, args):
        # set up params from regular densenet.
        super(AnytimeFCDensenet, self).__init__(args)
                
        # FC-dense specific
        self.n_pools = args.n_pools
        # other format is not supported yet
        assert self.n_pools * 2 + 1 == self.n_blocks
        # FC-dense doesn't support width > 1 yet
        assert self.width == 1
        # FC-dense doesn't like the starting pooling of imagenet initial conv/pool
        assert self.network_config.s_type == 'basic'

        # Precommpute the connection graph to figure out scales of each layer
        tmp = np.zeros(self.total_units + 1, dtype=int)
        cfg_cumsum = 1 + self.cumsum_blocks
        tmp[cfg_cumsum[:self.n_pools]] = 1
        tmp[cfg_cumsum[self.n_pools:-1]] = -1
        l_natural_scale = np.cumsum(tmp)
        l_min_scale = l_natural_scale.copy()
        l_max_scale = l_natural_scale.copy()
        ui= -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            for k in range(n_units):
                ui += 1
                indices = self.connections[ui]
                scale_ui = l_natural_scale[ui + 1]
                for idx in indices:
                    if l_min_scale[idx] > scale_ui:
                        l_min_scale[idx] = scale_ui
                    if l_max_scale[idx] < scale_ui:
                        l_max_scale[idx] = scale_ui
        self.l_natural_scale = l_natural_scale
        self.l_min_scale = l_min_scale
        self.l_max_scale = l_max_scale
        for i in range(self.total_units+1):
            logger.info("{} natural: {}  min : {}  max : {}".format(i, l_natural_scale[i],
                l_min_scale[i], l_max_scale[i]))


    def compute_transition_up(self, pls, skip_pls, trans_idx):
        """ for each previous layer, transition it up with deconv2d i.e., 
            conv2d_transpose
        """
        scale_bi = self.n_pools * 2 - 1 - trans_idx
        new_pls = []
        for pli, pl in enumerate(pls):
            if pli < len(skip_pls):
                new_pls.append(skip_pls[pli])
                continue 
            if self.l_min_scale[pli] > scale_bi:
                new_pls.append(None)
                continue
            # implied that skip_pls is exhausted
            with tf.variable_scope('transit_{:02d}_{:02d}'.format(trans_idx, pli)):
                ch_in = pl.get_shape().as_list()[CHANNEL_DIM]
                ch_out = int(ch_in * self.reduction_ratio)
                dyn_h = tf.shape(skip_pls[0])[HEIGHT_DIM]
                dyn_w = tf.shape(skip_pls[0])[WIDTH_DIM]
                #kernel_shape=3, stride=2
                new_pls.append(Deconv2D('deconv', pl, ch_out, 3, 2,
                    dyn_hw=[dyn_h, dyn_w], nl=BNReLU))
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
        growth = self.growth_rate
        for bi, n_units in enumerate(self.network_config.n_units_per_block):  
            pls, pmls = self.compute_block(pls, pmls, n_units, growth)
            if bi < self.n_pools: 
                # downsampling
                l_pls.append(pls)
                growth *= self.growth_rate_multiplier
                pls = self.compute_transition(pls, bi)
            elif bi < self.n_blocks - 1:
                # To check: first deconv at self.n_pools, every layer except the last block
                # has pl of featmap-size of n_pools-1. 
                # The second to the last block has the final upsampling, and it has 
                # bi = 2*n_pools - 1; The featmap scale matches that of l_pls[0] 
                skip_pls = l_pls[2*self.n_pools - bi - 1]
                growth /= self.growth_rate_multiplier
                pls = self.compute_transition_up(pls, skip_pls, bi)
        
        ll_feats = [ [ feat ] for feat in pmls ]
        assert len(ll_feats) == self.total_units
        return ll_feats


## Version 2 of anytime FCN for dense-net
# 
class AnytimeFCDensenetV2(AnytimeFCN, AnytimeLogDensenetV2):  
    
    def __init__(self, args):
        super(AnytimeFCDensenetV2, self).__init__(args)
        self.n_pools = args.n_pools
        assert self.n_pools * 2 == self.n_blocks - 1
        assert self.width == 1
        assert self.network_config.s_type == 'basic'

    def update_compressed_feature_up(self, layer_idx, ch_out, pls, bcml, sml):
        """
            layer_idx :
            pls : list of layers in the most recent block
            bcml : context feature for the most recent block
            sml : Final feature of the target scale on the down path 
                (it comes from skip connection)
        """
        with tf.variable_scope('transition_after_{}'.format(layer_idx)) as scope: 
            l = tf.concat(pls, CHANNEL_DIM, name='concat_new')
            #ch_new = l.get_shape().as_list()[CHANNEL_DIM]
            #ch_old = bcml.get_shape().as_list()[CHANNEL_DIM] 
            #ch_skip = sml.get_shape().as_list()[CHANNEL_DIM] 
            dyn_h = tf.shape(sml)[HEIGHT_DIM]
            dyn_w = tf.shape(sml)[WIDTH_DIM]
            l = Deconv2D('deconv_new', l, ch_out, 3, 2, 
                dyn_hw=[dyn_h,dyn_w], nl=BNReLU)
            bcml = Deconv2D('deconv_old', bcml, ch_out, 3, 2, 
                dyn_hw=[dyn_h,dyn_w], nl=BNReLU)
            bcml = tf.concat([sml, l, bcml], CHANNEL_DIM, name='concat_all')
        return bcml

    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        bcml = l_feats[0] #block compression merged layer
        l_mls = []
        growth = self.growth_rate
        layer_idx = -1
        l_skips = []
        for bi, n_units in enumerate(self.network_config.n_units_per_block):  
            pls, l_mls = self.compute_block(layer_idx, n_units, l_mls, bcml, growth)
            layer_idx += n_units
            ch_out = growth * (int(np.log2(self.total_units + 1)) + 1)
            if bi < self.n_pools: 
                l_skips.append(l_mls[-1])
                bcml = self.update_compressed_feature(layer_idx, ch_out, pls, bcml)
            elif bi < self.n_blocks - 1:
                sml = l_skips[self.n_pools * 2 - bi - 1]
                bcml = self.update_compressed_feature_up(layer_idx, ch_out, pls, bcml, sml)
        
        ll_feats = [ [ml] for ml in l_mls ]
        assert len(ll_feats) == self.total_units
        return ll_feats



###########################
# Multi-scale Dense-Network and its log-dense variant
###########################
def parser_add_msdensenet_arguments(parser):
    parser, depth_group = parser_add_common_arguments(parser)
    depth_group.add_argument('--msdensenet_depth',
                             help='depth of multiscale densenet', type=int)
    parser.add_argument('-g', '--growth_rate', help='growth rate at high resolution',
                        type=int, default=6)
    parser.add_argument('--bottleneck_width', help='multiplier of growth for width of bottleneck',
                        type=float, default=4.0)
    parser.add_argument('--num_scales', help='number of scales',
                        type=int, default=3)

class AnytimeMultiScaleDenseNet(AnytimeNetwork):
    
    def __init__(self, input_size, args):
        super(AnytimeMultiScaleDenseNet, self).__init__(input_size, args)
        self.num_scales = self.options.num_scales
        self.growth_rate = self.options.growth_rate
        self.bottleneck_width = self.options.bottleneck_width
        self.init_channel = self.growth_rate * 2

    def _compute_init_l_feats(self, image):
        l_feats = []
        ch_out = self.init_channel
        for w in range(self.num_scales):
            with tf.variable_scope('init_conv'+str(w)) as scope:
                if w == 0:
                    if self.network_config.s_type == 'basic':
                        l = Conv2D('conv0', image, ch_out, 3, nl=BNReLU) #, nl=BNReLU) 
                    else:
                        assert self.network_config.s_type == 'imagenet'
                        l = (LinearWrap(image)
                            .Conv2D('conv0', ch_out, 7, stride=2, nl=BNReLU)
                            .MaxPooling('pool0', shape=3, stride=2, padding='SAME')())
                else:
                    l = Conv2D('conv0', l, ch_out, 3, stride=2, nl=BNReLU) 
                l_feats.append(l)
                ch_out *= 2
        return l_feats

    def compute_edge(self, l, ch_out, l_type='normal', name=""):
        if self.network_config.b_type == 'bottleneck':
            bnw = int(self.bottleneck_width * ch_out)
            l = Conv2D('conv1x1_'+name, l, bnw, 1, nl=BNReLU)
        if l_type == 'normal':
            stride = 1
        elif l_type == 'down':
            stride = 2
        l = Conv2D('conv3x3_'+name, l, ch_out, 3, stride=stride, nl=BNReLU)
        return l
        

    def compute_block(self, bi, n_units, layer_idx, ll_merged_feats):
        l_mf = ll_merged_feats[-1]
        g_base = self.growth_rate * 2**bi
        for k in range(n_units):
            layer_idx += 1
            scope_name = self.compute_scope_basename(layer_idx)
            l_feats = [None] * bi
            g = g_base
            for w in range(bi, self.num_scales):
                with tf.variable_scope(scope_name+'.'+str(w)) as scope:
                    if w == bi and (layer_idx ==0 or k > 0):
                        l = self.compute_edge(l_mf[w], g, 'normal')
                    else:
                        l = self.compute_edge(l_mf[w], g/2, 'normal', name='e1')
                        lp = self.compute_edge(l_mf[w-1], g/2, 'down', name='e2')
                        l = tf.concat([l, lp], CHANNEL_DIM, name='concat_ms') 
                    l_feats.append(l)
                g *= 2
            #end for w
            new_l_mf = [None] * self.num_scales
            for w in range(bi, self.num_scales):
                with tf.variable_scope(scope_name+'.'+str(w)+'.merge') as scope:
                    new_l_mf[w] = tf.concat([l_mf[w], l_feats[w]], 
                        CHANNEL_DIM, name='merge_feats')
            ll_merged_feats.append(new_l_mf)
            l_mf = new_l_mf
        return ll_merged_feats


    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        ll_merged_feats = [[ feat for feat in l_feats ]]
        layer_idx = -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            ll_merged_feats = self.compute_block(bi, n_units, layer_idx, ll_merged_feats) 
            layer_idx += n_units
        ll_feats = [ [l_merged_feats[-1]] for l_merged_feats in ll_merged_feats ]
        # since ll_feats now contains the initial feature, which we don't want..
        return ll_feats[1:]
