import tensorflow as tf
import numpy as np
from .base import Callback
from ..tfutils import get_op_tensor_name, get_op_or_tensor_by_name
from tensorpack.utils import logger

__all__ = ['Exp3CPU', 'RWMCPU', 'FixedDistributionCPU', 'ThompsonSamplingCPU']

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

    def _after_run(self, select, reward):
        self._select = select
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
 
    def _before_run(self, _):
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

    def _after_run(self, *rewards):
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
 
    def _before_run(self, _):
        #print "fetch name: {}".format(self.rewards[self._select].name)
        return self.rewards


class FixedDistributionCPU(Callback):

    def __init__(self, K, select_name, distribution=None):
        self.K = K
        if distribution is not None:
            self.w = distribution / np.sum(distribution)
        else:
            self.w = np.ones(K, dtype=np.float64)
            if K >= 8:
                self.w = self.w / 2.0 / (K-4)
                self.w[K / 2] = 0.125
                self.w[K / 4] = 0.125
                self.w[K * 3 / 4 ] = 0.125
                self.w[K-1] = 0.125
            else:
                self.w = self.w / K

        # local record of selected value
        self._select = self.K - 1 
        self.select_name = select_name
        self._select_readable_name, self.select_var_name = get_op_tensor_name(select_name)

        self._select=self.K-1
        self.is_active=False

    def _setup_graph(self):
        all_vars = tf.global_variables()
        for v in all_vars:
            if v.name == self.select_var_name:
                self.select = v
                break
        else:
            raise ValueError("{} doesn't exist as VAR".format(self.select_var_name))
        self.select_holder = tf.placeholder(tf.int32, shape=(), name='selected_idx')
        self.assign_selection = self.select.assign(self.select_holder)
    
    def _before_train(self):
        pass

    def _after_run(self, select):
        if self.is_active:
            self._select = np.int32(np.argmax(np.random.multinomial(1, self.w)))
            self.assign_selection.eval(feed_dict={self.select_holder : self._select})

    def _trigger_epoch(self):
        self.is_active = True
 
    def _before_run(self, _):
        #print "fetch name: {}".format(self.rewards[self._select].name)
        return [self.select]

class ThompsonSamplingCPU(Callback):
    """
        Pseudo thompson sampling. Sample proportional to the reward of each option
    """

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

    def _after_run(self, *rewards):
        #print "select: {} , reward: {}".format(select, reward)
        rewards = np.asarray(rewards).reshape((self.K,))
        self.average_reward += rewards
        self.max_reward = np.maximum(rewards, self.max_reward)
        self.reward_cnt += 1
        if not self.is_first and self.active:
            self.w = self.w * 0.9 + rewards
            self.w = np.maximum(0, self.w)
            # bound away from 0
            assert not any(np.isnan(self.w)), self.w
        self.sample_w = self.w / np.sum(self.w)
        self.sample_w = self.sample_w * (1.0 - self.gamma) + self.gamma / self.K
        self._select = np.int32(np.argmax(np.random.multinomial(1, self.sample_w)))
        self.assign_selection.eval(feed_dict={self.select_holder : self._select})

        self.is_first = False

    def _trigger_epoch(self):
        self.active = True
        logger.info("Thompson: Average Reward: {}".format(self.average_reward / self.reward_cnt))
        logger.info("Thompson: Max Reward: {}".format(self.max_reward))
        logger.info("Thompson: Sample weights: {}".format(self.sample_w))
        logger.info("Thompson: weights: {}".format(self.w))
        self.old_average = self.average_reward
        self.average_reward = np.zeros(self.K)
        self.max_reward = np.zeros(self.K)
        self.reward_cnt = np.ones(self.K)
 
    def _before_run(self, _):
        #print "fetch name: {}".format(self.rewards[self._select].name)
        return self.rewards
