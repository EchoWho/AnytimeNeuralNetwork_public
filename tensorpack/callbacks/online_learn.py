import tensorflow as tf
import numpy as np
from .base import Callback
from ..tfutils import get_op_tensor_name, get_op_or_tensor_by_name
from tensorpack.utils import logger

__all__ = ['Exp3CPU', 'RWMCPU'] 

class Exp3CPU(Callback):

    def __init__(self, K, gamma, select_name, reward_names):
        assert K == len(reward_names)
        self.K = K
        self.gamma = gamma
        self.w = np.ones(K, dtype=np.float64) / K
        self.sample_w = np.ones(K, dtype=np.float64) / K
        # local record of selected value
        self._select = self.K - 1 
        self.select_name = select_name
        self._select_readable_name, self.select_var_name = get_op_tensor_name(select_name)
        self.reward_names = reward_names
        names = [get_op_tensor_name(name) for name in reward_names ]
        self._r_readable_names, self.r_names = zip(*names)
        self._r_readable_names = list(self._r_readable_names)
        self.r_names = list(self.r_names)

        self._select=self.K-1
        self.active = False
        self.is_first = True

    def _setup_graph(self):
        self.rewards = []
        all_vars = tf.global_variables()
        for v in all_vars:
            if v.name == self.select_var_name:
                self.select = v
                break
        else:
            raise ValueError("{} doesn't exist as VAR".format(self.select_var_name))
        for r_name in self.r_names:
            self.rewards.append(get_op_or_tensor_by_name(r_name))
        self.select_holder = tf.placeholder(tf.int32, shape=(), name='selected_idx')
        self.assign_selection = self.select.assign(self.select_holder)
    
    def _before_train(self):
        self.average_reward = np.zeros(self.K)
        self.max_reward = np.zeros(self.K)
        self.reward_cnt = np.ones(self.K)

    def _trigger_step(self, select, reward):
        assert select == self._select, select
        #print "select: {} , reward: {}".format(select, reward)
        self.average_reward[self._select] += reward
        self.max_reward[self._select] = max(reward, self.max_reward[self._select])
        self.reward_cnt[self._select] += 1
        if not self.is_first and self.active:
            old_weight = self.sample_w[self._select]
            self.w[self._select] *= np.exp(self.gamma * reward / (old_weight * self.K))
            self.w /= np.sum(self.w)
            self.w = self.w * (1.0 - self.gamma * 0.001) + self.gamma * 0.001 / self.K
            assert not any(np.isnan(self.w)), self.w
        self.sample_w = self.w * (1.0 - self.gamma) + self.gamma / self.K
        self._select = np.int32(np.argmax(np.random.multinomial(1, self.sample_w)))
        self.assign_selection.eval(feed_dict={self.select_holder : self._select})

        self.is_first = False

    def _trigger_epoch(self):
        self.active = True
        logger.info("Exp3: Average Reward: {}".format(self.average_reward / self.reward_cnt))
        logger.info("Exp3: Max Reward: {}".format(self.max_reward))
        logger.info("Exp3: Sample weights: {}".format(self.sample_w))
        logger.info("Exp3: weights: {}".format(self.w))
        self.old_average = self.average_reward
        self.average_reward = np.zeros(self.K)
        self.max_reward = np.zeros(self.K)
        self.reward_cnt = np.ones(self.K)
 
    def _extra_fetches(self):
        #print "fetch name: {}".format(self.rewards[self._select].name)
        return [self.select, self.rewards[self._select]]




class RWMCPU(Callback):

    def __init__(self, K, gamma, select_name, reward_names):
        assert K == len(reward_names)
        self.K = K
        self.gamma = gamma
        self.w = np.ones(K, dtype=np.float64) / K
        self.sample_w = np.ones(K, dtype=np.float64) / K
        # local record of selected value
        self._select = self.K - 1 
        self.select_name = select_name
        self._select_readable_name, self.select_var_name = get_op_tensor_name(select_name)
        self.reward_names = reward_names
        names = [get_op_tensor_name(name) for name in reward_names ]
        self._r_readable_names, self.r_names = zip(*names)
        self._r_readable_names = list(self._r_readable_names)
        self.r_names = list(self.r_names)

        self._select=self.K-1
        self.active = False
        self.is_first = True

    def _setup_graph(self):
        self.rewards = []
        all_vars = tf.global_variables()
        for v in all_vars:
            if v.name == self.select_var_name:
                self.select = v
                break
        else:
            raise ValueError("{} doesn't exist as VAR".format(self.select_var_name))
        for r_name in self.r_names:
            self.rewards.append(get_op_or_tensor_by_name(r_name))
        self.select_holder = tf.placeholder(tf.int32, shape=(), name='selected_idx')
        self.assign_selection = self.select.assign(self.select_holder)
    
    def _before_train(self):
        self.average_reward = np.zeros(self.K)
        self.max_reward = np.zeros(self.K)
        self.reward_cnt = np.ones(self.K)

    def _trigger_step(self, *rewards):
        #print "select: {} , reward: {}".format(select, reward)
        rewards = np.asarray(rewards).reshape((self.K,))
        self.average_reward += rewards
        self.max_reward = np.maximum(rewards, self.max_reward)
        self.reward_cnt += 1
        if not self.is_first and self.active:
            #self.w[self._select] *= np.exp(self.gamma * reward / (old_weight * self.K))
            self.w *= np.exp(self.gamma * 0.5 * rewards)
            self.w /= np.sum(self.w)
            # bound away from 0
            self.w = self.w * (1.0 - self.gamma * 0.001) + self.gamma * 0.001 / self.K
            assert not any(np.isnan(self.w)), self.w
        self.sample_w = self.w * (1.0 - self.gamma) + self.gamma / self.K
        self._select = np.int32(np.argmax(np.random.multinomial(1, self.sample_w)))
        self.assign_selection.eval(feed_dict={self.select_holder : self._select})

        self.is_first = False

    def _trigger_epoch(self):
        self.active = True
        logger.info("RWM: Average Reward: {}".format(self.average_reward / self.reward_cnt))
        logger.info("RWM: Max Reward: {}".format(self.max_reward))
        logger.info("RWM: Sample weights: {}".format(self.sample_w))
        logger.info("RWM: weights: {}".format(self.w))
        self.old_average = self.average_reward
        self.average_reward = np.zeros(self.K)
        self.max_reward = np.zeros(self.K)
        self.reward_cnt = np.ones(self.K)
 
    def _extra_fetches(self):
        #print "fetch name: {}".format(self.rewards[self._select].name)
        return self.rewards
