#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils import utils
from tensorpack.utils.stats import RatioCounter

from tensorpack.network_models import anytime_network
from tensorpack.network_models.anytime_network import AnytimeMultiScaleDenseNet

import get_augmented_data

args = None
INPUT_SIZE = 224
get_data = get_augmented_data.get_ilsvrc_augmented_data


def get_config():
    # prepare dataset
    dataset_train = get_data('train', args, do_multiprocess=True)
    dataset_val = get_data('val', args, do_multiprocess=True) 
    steps_per_epoch = dataset_train.size() // args.nr_gpu

    model=AnytimeMultiScaleDenseNet(INPUT_SIZE, args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()
    #lr_schedule = [(1, 1e-1/3), (30, 1e-2/3), (60, 1e-3/3), (90, 1e-4/3), (105, 1e-5/3)]

    lr_rate = args.lr_divider
    lr_schedule = [(1, 1e-1 / lr_rate), (60, 1e-2 / lr_rate ), (90, 1e-3 / lr_rate), (105, 1e-4 / lr_rate)]
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, keep_freq=10000),
            InferenceRunner(dataset_val, classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate'),
        ] + loss_select_cbs,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=128,
    )

def eval_on_ILSVRC12(subset):
    ds = get_data(subset, args, do_multiprocess=False)
    model = AnytimeMultiScaleDenseNet(INPUT_SIZE, args)

    output_names = []
    for i, w in enumerate(model.weights):
        if w > 0:
            output_names.append('layer{:03d}.0.pred/linear/output:0'.format(i))

    pred_config = PredictConfig(
        model=model,
        session_init=get_model_loader(args.load),
        input_names=['input', 'label'],
        output_names=output_names[-1:]
    )
    pred = SimpleDatasetPredictor(pred_config, ds)

    if args.store_basename is not None:
        store_fn = args.store_basename + "_{}.bin".format(subset)
        f_store_out = open(store_fn, 'wb')

    for o in pred.get_result():
        if args.store_basename is not None:
            preds = o[0]
            f_store_out.write(preds)

    if args.store_basename is not None:
        f_store_out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='ILSVRC dataset dir that contains the tf records directly')
    parser.add_argument('--log_dir', help='log_dir for stdout')
    parser.add_argument('--model_dir', help='dir for saving models')
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=128)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--evaluate', help='a comma separated list containing [train, test]',
                        default="", type=str)
    parser.add_argument('--store_basename', help='basename_<train/test>.bin for storing the logits',
                        type=str, default='None')
    anytime_network.parser_add_msdensenet_arguments(parser)
    args = parser.parse_args()
    
    # Fixed parameter
    args.ds_name="ilsvrc"
    args.num_classes == 1000
    args.init_channel = 16
    args.stack = (args.msdensenet_depth - 3) // 5

    # GPU will handle mean std transformation to save CPU-GPU communication
    args.do_mean_std_gpu_process = True
    args.input_type = 'uint8'
    args.mean = get_augmented_data.ilsvrc_mean
    args.std = get_augmented_data.ilsvrc_std
    assert args.do_mean_std_gpu_process and args.input_type == 'uint8'
    assert args.mean is not None and args.std is not None

    # Scale learning rate with the batch size linearly 
    args.lr_divider = 2.0 * 256.0 / args.batch_size 
    args.init_lr = 1e-1 / args.lr_divider
    args.batch_norm_decay=0.9**(2.0/args.lr_divider)  # according to Torch blog

    # directory setup
    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()

    args.evaluate = filter(bool, args.evaluate.split(','))
    do_eval = len(args.evaluate) > 0
    if do_eval:
        for subset in args.evaluate:
            if subset in ['train', 'val']:
                eval_on_ILSVRC12(subset)
        sys.exit()

    config = get_config()
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
