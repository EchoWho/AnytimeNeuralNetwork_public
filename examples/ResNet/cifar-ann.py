import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.models import Exp3,HalfEndHalfExp3,RandSelect,RWM
from tensorpack.utils import anytime_loss
from tensorpack.utils import logger
from tensorpack.utils import utils

from tensorpack.network_models.anytime_resnet import AnytimeResnet

"""
"""
NUM_RES_BLOCKS = 3
INPUT_SIZE=32

args=None

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    if args.num_classes == 10:
        ds = dataset.Cifar10(train_or_test, do_validation=args.do_validation)
    elif args.num_classes == 100:
        ds = dataset.Cifar100(train_or_test, do_validation=args.do_validation)
    else:
        raise ValueError('Number of classes must be set to 10(default) or 100 for CIFAR')
    if args.do_validation: 
        logger.info('[Validation] {} set has n_samples: {}'.format(isTrain, len(ds.data)))
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, args.batch_size, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_config():
    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    # Generate model loss weights
    total_units = AnytimeResnet.compute_total_units(NUM_RES_BLOCKS,
        args.num_units, args.width)
    weights = anytime_loss.loss_weights(total_units, args)
    logger.info('weights: {}'.format(weights))
    
    # Specify model
    model=AnytimeResnet(args.num_units, args.width, args.init_channel, 
        args.num_classes, weights, INPUT_SIZE, args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    lr_schedule = [(1, 0.1), (82, 0.01), (123, 0.001), (250, 0.0002)]
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost')] + classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule)
        ] + loss_select_cbs,
        model=model,
        steps_per_epoch=steps_per_epoch,
        max_epoch=300,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
    parser.add_argument('--model_dir', help='model_dir position',
                        type=str, default=None)

    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=128)
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=5)
    parser.add_argument('-w', '--width',
                        help='width of the network',
                        type=int, default=1)
    parser.add_argument('-c', '--init_channel',
                        help='channel at beginning of each width of the network',
                        type=int, default=16)
    parser.add_argument('-s', '--stack', 
                        help='number of units per stack, \
                              i.e., number of units per prediction',
                        type=int, default=1)
    parser.add_argument('--num_classes', help='Number of classes', 
                        type=int, default=10)
    parser.add_argument('--stop_gradient', help='Whether to stop gradients.',
                        type=bool, default=False)
    parser.add_argument('--sg_gamma', help='Gamma for partial stop_gradient',
                        type=np.float32, default=0)
    parser.add_argument('--samloss', 
                        help='Method to Sample losses to update',
                        type=int, default=0)
    parser.add_argument('--exp_gamma', help='Gamma for exp3 in sample loss',
                        type=np.float32, default=0.3)
    parser.add_argument('--sum_rand_ratio', help='frac{Sum weight}{randomly selected weight}',
                        type=np.float32, default=2.0)
    parser.add_argument('--last_reward_rate', help='rate of last reward in comparison to the max',
                        type=np.float32, default=0.85)
    parser.add_argument('--do_validation', help='Whether use validation set. Default not',
                        type=bool, default=False)
    parser.add_argument('-f', '--func_type', 
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr', 
                        type=int, default=5)
    parser.add_argument('--exponential_base', help='Exponential base',
                        type=np.float32)
    parser.add_argument('--opt_at', help='Optimal at', 
                        type=int, default=-1)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()
    fs.set_dataset_path(path=args.data_dir, auto_download=False)

    config = get_config()
    if args.load and os.path.exists(arg.load):
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
