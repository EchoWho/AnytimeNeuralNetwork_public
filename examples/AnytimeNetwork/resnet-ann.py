import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
import ipdb as pdb
import struct 
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils import utils

from tensorpack.network_models import anytime_network
from tensorpack.network_models.anytime_network import AnytimeResnet

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


def evaluate(model_cls, ds, eval_names):
    assert args is not None, args
    model = model_cls(INPUT_SIZE, args) 

    output_names = []
    for i, w in enumerate(model.weights):
        if w > 0:
            output_names.append('layer{:03d}.0.pred/linear/output:0'.format(i))

    pred_config = PredictConfig(
        model=model,
        session_init=SaverRestore(args.load),
        input_names=['input', 'label'],
        output_names=['input', 'label'] + output_names)
    
    pred = SimpleDatasetPredictor(pred_config, ds)

    if args.store_final_prediction:
        store_fn = args.store_basename + "_{}.bin".format(eval_name)
        f_store_out = open(store_fn, 'wb')

    l_labels = []
    for idx, output in enumerate(pred.get_result()):
        # o contains a list of predictios at various locations; each pred contains a small batch
        image, label = output[0:2]
        l_labels.extend(label)
        anytime_preds = output[2:]
        
        if args.store_final_prediction:
            preds = anytime_preds[-1]
            f_store_out.write(preds)

    # since the labels comes in batches
    l_labels = np.asarray(l_labels)
    logger.info("N samples predicted: {}".format(len(l_labels)))
    
    if args.store_final_prediction:
        f_store_out.close()
        
        ## report accuracy of the stored predicitons
        with open(store_fn, 'rb') as fin:
            n_wrong = 0
            row_len = 4 * args.num_classes
            for label in l_labels:
                contents = fin.read(row_len)
                logit = np.asarray(struct.unpack('f'*args.num_classes, contents)).reshape([1, args.num_classes])
                n_wrong += int(np.argmax(logit, axis=1) != label)

        error_rate = n_wrong / np.float32(len(l_labels))
        logger.info("Verify error rate of the stored prediction to be {}".format(error_rate))


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
    parser.add_argument('--evaluate', help='a comma separated list containing [train, test]',
                        default="", type=str)
    parser.add_argument('--store_final_prediction', help='wheter evaluation stores final prediction',
                        default=False, action='store_true')
    parser.add_argument('--store_basename', help='basename_<train/test>.bin for storing the logits',
                        type=str, default='distill_target')
    anytime_network.parser_add_resnet_arguments(parser)
    model_cls = AnytimeResnet
    args = parser.parse_args()

    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()
    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))

    # generate a list of none-empty strings for specifying the splits
    args.evaluate = filter(bool, args.evaluate.split(','))
    do_eval = len(args.evaluate) > 0

    ## Set dataset-network specific assert/info
    if args.ds_name == 'cifar10' or args.ds_name == 'cifar100':
        if args.ds_name == 'cifar10':
            args.num_classes = 10
        else:
            args.num_classes = 100
        args.regularize_coef = 'decay'
        INPUT_SIZE = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = get_cifar_augmented_data
        ds_train = get_data('train', args, not do_eval)
        ds_val = get_data('test', args, False)

        lr_schedule = \
            [(1, 0.1), (82, 0.01), (123, 0.001), (250, 0.0002)]
        max_epoch = 300

        if do_eval:
            for eval_name in args.evaluate:
                if eval_name == 'train':
                    ds = ds_train
                elif eval_name == 'test':
                    ds = ds_val
                evaluate(model_cls, ds, eval_name)
            sys.exit()

    elif args.ds_name == 'svhn':
        args.num_classes = 10
        args.regularize_coef = 'decay'
        INPUT_SIZE = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = get_svhn_augmented_data
        
        if do_eval:
            if 'train' in args.evaluate:
                args.evaluate.append('extra')
            for eval_name in args.evaluate:
                ds = get_data(eval_name, args, do_multiprocess=False)
                evaluate(model_cls, ds, eval_name)
            sys.exit()

        ## Training model 
        ds_train = get_data('train', args, do_multiprocess=True)
        ds_val = get_data('test', args, do_multiprocess=False)

        lr_schedule = \
            [(1, 0.1), (15, 0.01), (30, 0.001), (45, 0.0002)]
        max_epoch = 60


    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
