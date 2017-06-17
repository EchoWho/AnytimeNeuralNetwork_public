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
NUM_UNITS = 5
WIDTH = 1
INIT_CHANNEL = 16
NUM_CLASSES = 10

# anytime loss skip (num units per stack/prediction)
NUM_UNITS_PER_STACK=1

# Random loss sample params
##0: nothing; 1: rand; 2:exp3; 3:HEHE3
SAMLOSS=0  
EXP3_GAMMA=0.5
SUM_RAND_RATIO=2.0
LAST_REWARD_RATE=0.85

# Stop gradients params
STOP_GRADIENTS=False
STOP_GRADIENTS_PARTIAL=False
SG_GAMMA = 0.3

class AnytimeResnet(ModelDesc):

    def __init__(self, n, width, init_channel, num_classes, weights):
        super(Model, self).__init__()
        self.n = n
        self.width = width
        self.init_channel = init_channel
        self.num_classes = num_classes
        self.weights = weights
        self.select_idx_name = "select_idx"

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]
    
    def compute_scope_basename(self, layer_idx):
        return "layer{:03d}".format(layer_idx)

    def compute_classification_callbacks(self):
        vcs = []
        total_units = NUM_RES_BLOCKS * self.n * self.width
        unit_idx = -1
        layer_idx=-1
        for bi in range(NUM_RES_BLOCKS):
            for k in range(self.n):
                layer_idx += 1
                for wi in range(self.width):
                    unit_idx += 1
                    weight = self.weights[unit_idx]
                    if weight > 0:
                        scope_name = compute_scope_basename(layer_idx)
                        scope_name += '.'+str(wi)+'.pred/' 
                        vcs.append(ClassificationError(\
                            wrong_tensor_name=scope_name+'incorrect_vector:0', 
                            summary_name=scope_name+'val_err'))
        return vcs

    def compute_loss_select_callbacks(self):
        if SAMLOSS > 0:
            weights = self.weights
            ls_K = np.sum(np.asarray(weights) > 0)
            reward_names = [ 'tower0/reward_{:02d}:0'.format(i) for i in range(ls_K)]
            select_idx_name = '{}:0'.format(self.select_idx_name)
            if SAMLOSS == 3:
                online_learn_cb = FixedDistributionCPU(ls_K, select_idx_name, None)
            elif SAMLOSS == 6:
                online_learn_cb = FixedDistributionCPU(ls_K, select_idx_name, 
                    weights[weights>0])
            else:    
                gamma = EXP3_GAMMA
                if SAMLOSS == 1:
                    online_learn_func = Exp3CPU
                    gamma = 1.0
                elif SAMLOSS == 2:
                    online_learn_func = Exp3CPU
                elif SAMLOSS == 4:
                    online_learn_func = RWMCPU
                elif SAMLOSS == 5:
                    online_learn_func = ThompsonSamplingCPU
                online_learn_cb = online_learn_func(ls_K, gamma, 
                    select_idx_name, reward_names)
            online_learn_cbs = [ online_learn_cb ]
        else:
            online_learn_cbs = []
        return online_learn_cbs

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.transpose(image, [0,3,1,2])

        def residual(name, l_feats, increase_dim=False):
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

        def predictions_and_losses(name, l_feats, out_dim, label):
            l_logits = []
            var_list = []
            l_costs = []
            l_wrong = []
            for w in range(self.width):
                with tf.variable_scope(name+'.'+str(w)+'.pred') as scope:
                    l = tf.nn.relu(l_feats[w])
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

                    wrong = prediction_incorrect(logits, label)
                    wrong = tf.reduce_mean(wrong, name='train_error')
                    add_moving_summary(wrong)

                    l_costs.append(cost)
                    l_wrong.append(wrong)

            return l_logits, var_list, l_costs, l_wrong

        logger.info("sampling loss with method {}".format(SAMLOSS))
        ls_K = np.sum(np.asarray(self.weights) > 0)
        if SAMLOSS > 0:
            select_idx = tf.get_variable(self.select_idx_name, (), tf.int32,
                initializer=tf.constant_initializer(ls_K - 1), trainable=False)
            tf.summary.scalar(self.select_idx_name, select_idx)
            for i in range(ls_K):
                weight_i = tf.cast(tf.equal(select_idx, i), tf.float32, 
                                   name='weight_{}'.format(i))
                add_moving_summary(weight_i)

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], 
                      data_format='NCHW'), \
             argscope(Conv2D, kernel_shape=3, nl=tf.identity, use_biase=False, 
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
            if STOP_GRADIENTS_PARTIAL:
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
                    scope_name = compute_scope_basename(layer_idx)
                    l_feats = residual(scope_name, l_feats, \
                                       increase_dim=(k==0 and res_block_i > 0))
                    ll_feats.append(l_feats)

                    # In case that we need to stop gradients
                    is_last_row = res_block_i == NUM_RES_BLOCKS-1 and k==self.n-1
                    if STOP_GRADIENTS_PARTIAL and not is_last_row:
                        l_new_feats = []
                        for fi, f in enumerate(l_feats):
                            unit_idx +=1
                            if self.weights[unit_idx] > 0:
                                f = (1-SG_GAMMA)*tf.stop_gradient(f) + SG_GAMMA*f
                            l_new_feats.append(f)
                        l_feats = l_new_feats
                # end for each k in self.n
            #end for each block
            
            unit_idx = -1
            for layer_idx, l_feats in enumerate(ll_feats)
                scope_name = compute_scope_basename(layer_idx)
                l_logits, var_list, l_costs, l_wrong = predictions_and_losses(\
                    scope_name, l_feats, self.num_classes, label)
                    
                for ci, c in enumerate(l_costs):
                    unit_idx += 1
                    cost_weight = self.weights[unit_idx]
                    if cost_weight > 0:
                        anytime_idx += 1

                        # Additional weight for unit_idx. 
                        add_weight = 0
                        if SAMLOSS > 0:
                            add_weight = tf.cond(tf.equal(anytime_idx, 
                                                          select_idx),
                                lambda: tf.constant(self.weights[-1] * 2.0, 
                                                    dtype=tf.float32),
                                lambda: tf.constant(0, dtype=tf.float32))
                        if SUM_RAND_RATIO > 0:
                            cost += (cost_weight + add_weight / SUM_RAND_RATIO) * c
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
                            reward = max_reward * LAST_REWARD_RATE
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
        add_moving_summary(cost, wd_cost)
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt
