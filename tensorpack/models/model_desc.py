#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import pickle
import six

from ..utils import logger
from ..utils.naming import INPUTS_KEY
from ..utils.argtools import memoized
from ..tfutils.model_utils import apply_slim_collections

__all__ = ['InputDesc', 'InputVar', 'ModelDesc', 'ModelFromMetaGraph']


class InputDesc(object):
    """ Store metadata about input placeholders. """
    def __init__(self, type, shape, name, sparse=False):
        """
        Args:
            type: tf type of the tensor.
            shape (list):
            name (str):
            sparse (bool): whether to use ``tf.sparse_placeholder``.
        """
        self.type = type
        self.shape = shape
        self.name = name
        self.sparse = sparse

    def dumps(self):
        return pickle.dumps(self)

    @staticmethod
    def loads(buf):
        return pickle.loads(buf)


class InputVar(InputDesc):
    def __init__(self, *args, **kwargs):
        logger.warn("[Deprecated] InputVar was renamed to InputDesc!")
        super(InputVar, self).__init__(*args, **kwargs)


@six.add_metaclass(ABCMeta)
class ModelDesc(object):
    """ Base class for a model description.
    """

# inputs:
    @memoized
    def get_reused_placehdrs(self):
        """
        Create or return (if already created) raw input TF placeholders in the graph.

        Returns:
            list[tf.Tensor]: the list of input placeholders in the graph.
        """
        return self.build_placeholders()

    def build_placeholders(self, prefix=''):
        """
        For each InputDesc, create new placeholders with optional prefix and
        return them. Useful when building new towers.

        Returns:
            list[tf.Tensor]: the list of built placeholders.
        """
        inputs = self._get_inputs()
        for v in inputs:
            tf.add_to_collection(INPUTS_KEY, v.dumps())
        ret = []
        with tf.name_scope(None):   # clear any name scope it might get called in
            for v in inputs:
                placehdr_f = tf.placeholder if not v.sparse else tf.sparse_placeholder
                ret.append(placehdr_f(
                    v.type, shape=v.shape,
                    name=prefix + v.name))
        return ret

    def get_inputs_desc(self):
        """
        Returns:
            list[:class:`InputDesc`]: list of the underlying :class:`InputDesc`.
        """
        return self._get_inputs()

    @abstractmethod
    def _get_inputs(self):
        """
        :returns: a list of InputDesc
        """

    def build_graph(self, model_inputs):
        """
        Build the whole symbolic graph.

        Args:
            model_inputs (list[tf.Tensor]): a list of inputs, corresponding to
                InputDesc of this model.
        """
        self._build_graph(model_inputs)

    @abstractmethod
    def _build_graph(self, inputs):
        pass

    def get_cost(self):
        """
        Return the cost tensor in the graph.
        Used by some of the tensorpack :class:`Trainer` which assumes single-cost models.
        You can ignore this method if you use your own trainer with more than one cost.

        It calls :meth:`ModelDesc._get_cost()` which by default returns
        ``self.cost``. You can override :meth:`_get_cost()` if needed.

        This function also applies the collection
        ``tf.GraphKeys.REGULARIZATION_LOSSES`` to the cost automatically.
        Because slim users would expect the regularizer being automatically applied once used in slim layers.
        """
        cost = self._get_cost()
        return apply_slim_collections(cost)

    def _get_cost(self, *args):
        return self.cost

    @memoized
    def get_optimizer(self):
        """
        Return the optimizer used in the task.
        Used by some of the tensorpack :class:`Trainer` which assume single optimizer.
        You can (and should) ignore this method if you use a custom trainer with more than one optimizers.

        Users of :class:`ModelDesc` will need to implement `_get_optimizer()`,
        which will only be called once per each model.

        Returns:
            a :class:`tf.train.Optimizer` instance.
        """
        return self._get_optimizer()

    def _get_optimizer(self):
        raise NotImplementedError()

    def get_gradient_processor(self):
        return self._get_gradient_processor()

    def _get_gradient_processor(self):
        return []


class ModelFromMetaGraph(ModelDesc):
    """
    Load the exact TF graph from a saved meta_graph.
    Only useful for inference.
    """

    # TODO this class may not be functional anymore.

    def __init__(self, filename):
        """
        Args:
            filename (str): file name of the saved meta graph.
        """
        tf.train.import_meta_graph(filename)
        all_coll = tf.get_default_graph().get_all_collection_keys()
        for k in [INPUTS_KEY, tf.GraphKeys.TRAINABLE_VARIABLES,
                  tf.GraphKeys.GLOBAL_VARIABLES]:
            if k not in all_coll:
                logger.warn("Collection {} not found in metagraph!".format(k))

    def _get_inputs(self):
        col = tf.get_collection(INPUTS_KEY)
        col = [InputDesc.loads(v) for v in col]
        return col

    def _build_graph(self, _, __):
        """ Do nothing. Graph was imported already """
        pass
