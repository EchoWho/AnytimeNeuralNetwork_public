import tensorflow as tf
import numpy as np

from ..tfutils.tower import get_current_tower_context
from ..utils import logger

__all__ = ['Exp3']

class Exp3(object):
    def __init__(self, name, K, gamma):
        with tf.variable_scope(name) as scope:
            self.name = name
            self.K = K
            self.gamma = gamma
            self.w = tf.get_variable("w", [1, self.K], 
                initializer=tf.constant_initializer(1.0/self.K), trainable=False)

    def sample(self):
        ctx = get_current_tower_context()
        if ctx.is_training is not None and not ctx.is_training:
            return -1, 0.0

        with tf.variable_scope(self.name):
            probs = (1.0 - self.gamma) * self.w + self.gamma / self.K
            idx = tf.cast(tf.multinomial(tf.log(probs), 1)[0][0], tf.int32)
            p_idx = probs[idx]
            return idx, p_idx

    def update(self, idx, p_idx, reward):
        ctx = get_current_tower_context()
        if ctx.is_training is not None and not ctx.is_training:
            return
            
        with tf.variable_scope(self.name):
            reward = reward / p_idx
            r_vec = tf.one_hot(idx, self.K, tf.exp(self.gamma * reward /self.K), 1.0)
            self.w = tf.multiply(self.w, r_vec)
            w_sum = tf.reduce_sum(self.w)
            self.w = self.w / w_sum
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.identity(self.w))
