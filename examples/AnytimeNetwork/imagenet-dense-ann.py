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
from tensorpack.network_models.anytime_network import \
    AnytimeDensenet, DenseNet, AnytimeLogDensenetV2, AnytimeLogLogDenseNet

import get_augmented_data

args = None
INPUT_SIZE = 224

get_data = get_augmented_data.get_ilsvrc_augmented_data

def get_config(model_cls, lr_schedule):
    # prepare dataset
    dataset_train = get_data('train', args, do_multiprocess=True)
    dataset_val = get_data('val', args, do_multiprocess=True)
    steps_per_epoch = dataset_train.size() // args.nr_gpu

    model=model_cls(INPUT_SIZE, args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, keep_freq=12),
            InferenceRunner(dataset_val, classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate'),
        ] + loss_select_cbs,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )

def eval_on_ILSVRC12(model_file, data_dir):
    ds = get_data('val')
    model = AnytimeResnet(INPUT_SIZE, args)
    pred_config = PredictConfig(
        model=model,
        session_init=get_model_loader(model_file),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='ILSVRC dataset dir that contains the tf records directly')
    parser.add_argument('--log_dir', help='log_dir for stdout')
    parser.add_argument('--model_dir', help='dir for saving models')
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=128)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--is_toy', help='Whether to have data size of only 1024',
                        type=bool, default=False)
    anytime_network.parser_add_densenet_arguments(parser)
    args = parser.parse_args()
    args.ds_name="ilsvrc"
    if args.densenet_version == 'atv1':
        model_cls = AnytimeDensenet
    elif args.densenet_version == 'atv2':
        model_cls = AnytimeLogDensenetV2
    elif args.densenet_version == 'dense':
        model_cls = DenseNet
        args.reduction_ratio = 0.5
    elif args.densenet_version == 'loglog':
        model_cls = AnytimeLogLogDenseNet

    logger.info("Model class is {}".format(model_cls))

    # ilsvrc has 1000 classes
    args.num_classes = 1000

    # GPU will handle mean std transformation to save CPU-GPU communication
    args.do_mean_std_gpu_process = True
    args.input_type = 'uint8'
    args.mean = get_augmented_data.ilsvrc_mean
    args.std = get_augmented_data.ilsvrc_std
    assert args.do_mean_std_gpu_process and args.input_type == 'uint8'
    assert args.mean is not None and args.std is not None

    # Scale learning rate with the batch size linearly 
    # lr_rate represent how much we need to scale up down lr based on batch size
    lr_rate = args.batch_size / 256.0
    lr_decay_epoch = 30
    # step**6 approx 0.005 and step**5 approx 0.01
    lr_decay = 0.1 
    curr_lr = 0.1 * lr_rate
    max_epoch = 120
    lr_schedule = [] #(1, 1e-1 * lr_rate)]
    for epochi in range(1, max_epoch+1):
        lr_schedule.append( (epochi, curr_lr) ) 
        if epochi % lr_decay_epoch == 0:
            curr_lr *= lr_decay
    print lr_schedule
    args.init_lr = lr_schedule[0][1]
    args.batch_norm_decay=0.9**(lr_rate)  # according to Torch blog

    # directory setup
    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()

    config = get_config(model_cls, lr_schedule)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
