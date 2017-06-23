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
from tensorpack.network_models.anytime_network import AnytimeDensenet

"""
"""
INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None

def get_cifar_data(train_or_test):
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
    ds = BatchData(ds, args.batch_size // args.nr_gpu, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_svhn_data(train_or_test):
    isTrain = train_or_test == 'train'
    pp_mean = dataset.SVHNDigit.get_per_pixel_mean()
    if isTrain:
        d1 = dataset.SVHNDigit('train')
        d2 = dataset.SVHNDigit('extra')
        ds = RandomMixData([d1, d2])
    else:
        ds = dataset.SVHNDigit('test')

    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.Brightness(10),
            imgaug.Contrast((0.8, 1.2)),
            imgaug.GaussianDeform(  # this is slow. without it, can only reach 1.9% error
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (40, 40), 0.2, 3),
            imgaug.RandomCrop((32, 32)),
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, args.batch_size // args.nr_gpu, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
    return ds


def get_ilsvrc12_tfrecord_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.ILSVRC12TFRecord(args.data_dir, 
                                  train_or_test, 
                                  args.batch_size // args.nr_gpu, 
                                  height=INPUT_SIZE, 
                                  width=INPUT_SIZE)
    return ds


def get_config(ds_trian, ds_val, model_cls):
    # prepare dataset
    steps_per_epoch = ds_train.size() // args.nr_gpu

    model=model_cls(INPUT_SIZE, args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, keep_freq=12),
            InferenceRunner(ds_val,
                            [ScalarStats('cost')] + classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate')
        ] + loss_select_cbs,
        model=model,
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )


#def eval_on_ILSVRC12(model_file, data_dir):
#    ds = get_data('val')
#    model = AnytimeResnet(INPUT_SIZE, args)
#    pred_config = PredictConfig(
#        model=model,
#        session_init=get_model_loader(model_file),
#        input_names=['input', 'label'],
#        output_names=['wrong-top1', 'wrong-top5']
#    )
#    pred = SimpleDatasetPredictor(pred_config, ds)
#    acc1, acc5 = RatioCounter(), RatioCounter()
#    for o in pred.get_result():
#        batch_size = o[0].shape[0]
#        acc1.feed(o[0].sum(), batch_size)
#        acc5.feed(o[1].sum(), batch_size)
#    print("Top1 Error: {}".format(acc1.ratio))
#    print("Top5 Error: {}".format(acc5.ratio))

#if args.eval:
#    BATCH_SIZE = 128    # something that can run on one gpu
#    eval_on_ILSVRC12(args.load, args.data_dir)
#    sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset choice 
    parser.add_argument('--ds_name', help='name of dataset',
                        type=str, 
                        choices=['cifar10', 'cifar100', 'svhn', 'imagenet'])
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
    parser = anytime_network.parser_add_densenet_arguments(parser)
    model_cls = AnytimeDensenet
    args = parser.parse_args()
    assert args.batch_size <= 64

    ## Set dataset-network specific assert/info
    if args.ds_name == 'cifar10' or args.ds_name == 'cifar100':
        if args.ds_name == 'cifar10':
            args.num_classes = 10
        else:
            args.num_classes = 100
        INPUT_SIZE = 32
        get_data = get_cifar_data
        ds_train = get_data('train')
        ds_val = get_data('test')
        fs.set_dataset_path(path=args.data_dir, auto_download=False)

        lr_schedule = \
            [(1, 0.1), (150, 0.01), (225, 0.001)]
        max_epoch=300


    elif args.ds_name == 'svhn':
        args.num_classes = 10
        INPUT_SIZE = 32
        get_data = get_svhn_data
        ds_train = get_data('train')
        ds_val = get_data('test')
        fs.set_dataset_path(path=args.data_dir, auto_download=False)

        lr_schedule = \
            [(1, 0.1), (20, 0.01), (30, 0.001), (45, 0.0001)]
        max_epoch = 60


    elif args.ds_name == 'imagenet':
        args.num_classes = 1000
        INPUT_SIZE = 224
        get_data = get_ilsvrc12_tfrecord_data
        if args.is_toy:
            ds_train = get_data('toy_train')
            ds_val = get_data('toy_validation')
        else:
            ds_train = get_data('train')
            ds_val = get_data('validation')

        lr_schedule = \
            [(1, 0.05), (30, 0.01), (60, 1e-3), (85, 1e-4), (100, 1e-5)]
        max_epoch=115
         
    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))

    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()

    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(arg.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
