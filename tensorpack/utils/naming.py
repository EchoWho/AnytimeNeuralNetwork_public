# -*- coding: UTF-8 -*-
# File: naming.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

GLOBAL_STEP_INCR_OP_NAME = 'global_step_incr'
GLOBAL_STEP_INCR_VAR_NAME = 'global_step_incr:0'

LOCAL_STEP_OP_NAME = 'local_step'
LOCAL_STEP_VAR_NAME = 'local_step:0'

# prefix of predict tower
PREDICT_TOWER = 'towerp'

# extra variables to summarize during training in a moving-average way
MOVING_SUMMARY_OPS_KEY = 'MOVING_SUMMARY_OPS'

# metainfo for input tensors
INPUTS_KEY = 'INPUTS_METAINFO'

SUMMARY_BACKUP_KEYS = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_OPS_KEY]

TOWER_FREEZE_KEYS = SUMMARY_BACKUP_KEYS + [tf.GraphKeys.UPDATE_OPS]

# export all upper case variables
all_local_names = locals().keys()
__all__ = [x for x in all_local_names if x.isupper()]
