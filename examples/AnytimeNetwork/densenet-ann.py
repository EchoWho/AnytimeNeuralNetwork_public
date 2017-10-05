import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils import utils

from tensorpack.network_models import anytime_network
from tensorpack.network_models.anytime_network import \
AnytimeDensenet, DenseNet, AnytimeLogDensenetV2, AnytimeLogLogDenseNet


from get_augmented_data import get_cifar_augmented_data, get_svhn_augmented_data

"""
"""
INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None


def get_config(ds_trian, ds_val, model_cls):
    # prepare dataset
    steps_per_epoch = ds_train.size() // args.nr_gpu

    model=model_cls(INPUT_SIZE, args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, keep_freq=10000),
            InferenceRunner(ds_val,
                            [ScalarStats('cost')] + classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate')
        ] + loss_select_cbs,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset choice 
    parser.add_argument('--ds_name', help='name of dataset',
                        type=str, 
                        choices=['cifar10', 'cifar100', 'svhn'])
    # other common args
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=128)
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
    parser.add_argument('--model_dir', help='model_dir position',
                        type=str, default=None)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--do_validation', help='Whether use validation set. Default not',
                        type=bool, default=False)
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--is_toy', help='Whether to have data size of only 1024',
                        type=bool, default=False)
    anytime_network.parser_add_densenet_arguments(parser)
    args = parser.parse_args()
    lr_multiplier = 1.0 * args.batch_size / 64
    if args.densenet_version == 'atv1':
        model_cls = AnytimeDensenet
        lr_multiplier *= 1
    elif args.densenet_version == 'atv2':
        model_cls = AnytimeLogDensenetV2
        lr_multiplier *= 2.0
    elif args.densenet_version == 'dense':
        model_cls = DenseNet
        lr_multiplier *= 1
    elif args.densenet_version == 'loglog':
        model_cls = AnytimeLogLogDenseNet
        lr_multiplier *= 1

    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()
    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))

    assert args.batch_size <= 64

    ## Set dataset-network specific assert/info
    if args.ds_name == 'cifar10' or args.ds_name == 'cifar100':
        if args.ds_name == 'cifar10':
            args.num_classes = 10
        else:
            args.num_classes = 100
        INPUT_SIZE = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = get_cifar_augmented_data
        ds_train = get_data('train', args, do_multiprocess=True)
        ds_val = get_data('test', args, do_multiprocess=False)

        lr_schedule = [(1, 0.1), (140, 0.01), (210, 0.001)]
        lr_schedule = [ (t, v*lr_multiplier) for (t, v) in lr_schedule ] 
        max_epoch=250


    elif args.ds_name == 'svhn':
        args.num_classes = 10
        INPUT_SIZE = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = get_svhn_augmented_data
        ds_train = get_data('train', args, do_multiprocess=True)
        ds_val = get_data('test', args, do_multiprocess=False)

        lr_schedule = [(1, 0.1), (20, 0.01), (30, 0.001), (45, 0.0001)]
        lr_schedule = [ (t, v*lr_multiplier) for (t, v) in lr_schedule ] 
        max_epoch = 60
         
    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
