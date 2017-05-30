# -*- coding: UTF-8 -*-
# File: model_utils.py
# Author: tensorpack contributors

import tensorflow as tf
from termcolor import colored
from tabulate import tabulate

from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from .summary import add_moving_summary

__all__ = ['describe_model', 'get_shape_str', 'apply_slim_collections']


def describe_model():
    """ Print a description of the current model parameters """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if len(train_vars) == 0:
        logger.info("No trainable variables in the graph!")
        return
    total = 0
    data = []
    for v in train_vars:
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        data.append([v.name, shape.as_list(), ele])
    table = tabulate(data, headers=['name', 'shape', 'dim'])
    size_mb = total * 4 / 1024.0**2
    summary_msg = colored(
        "\nTotal #vars={}, #param={} ({:.02f} MB assuming all float32)".format(
            len(data), total, size_mb), 'cyan')
    logger.info(colored("Model Parameters: \n", 'cyan') + table + summary_msg)


def get_shape_str(tensors):
    """
    Args:
        tensors (list or tf.Tensor): a tensor or a list of tensors
    Returns:
        str: a string to describe the shape
    """
    if isinstance(tensors, (list, tuple)):
        for v in tensors:
            assert isinstance(v, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(v))
        shape_str = ",".join(
            map(lambda x: str(x.get_shape().as_list()), tensors))
    else:
        assert isinstance(tensors, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(tensors))
        shape_str = str(tensors.get_shape().as_list())
    return shape_str


def apply_slim_collections(cost):
    """
    Add the cost with the regularizers in ``tf.GraphKeys.REGULARIZATION_LOSSES``.

    Args:
        cost: a scalar tensor

    Return:
        a scalar tensor, the cost after applying the collections.
    """
    regulization_losses = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    ctx = get_current_tower_context()
    if len(regulization_losses) > 0:
        assert not ctx.has_own_variables, "REGULARIZATION_LOSSES collection doesn't work in replicated mode!"
        logger.info("Applying REGULARIZATION_LOSSES on cost.")
        reg_loss = tf.add_n(list(regulization_losses), name="regularize_loss")
        cost = tf.add(reg_loss, cost, name='total_cost')
        add_moving_summary(reg_loss, cost)
    return cost
