import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils import utils
from tensorpack.utils import fs

from tensorpack.network_models import anytime_network
from tensorpack.network_models.anytime_network import FCDensenet


"""
"""
INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None

def get_camvid_data(which_set, shuffle=True, slide_all=False):
    isTrain = which_set == 'train' or which_set == 'trainval'

    side = 224
    pixel_z_normalize = True 
    ds = dataset.Camvid(which_set, shuffle=shuffle, 
        pixel_z_normalize=pixel_z_normalize,
        slide_all=False,
        slide_window_size=side,
        void_overlap=not isTrain)
    if isTrain:
        x_augmentors = []
        xy_augmentors = [
            imgaug.RandomCrop((224, 224)),
            imgaug.Flip(horiz=True),
        ]
    else:
        x_augmentors = []
        xy_augmentors = [
            imgaug.RandomCrop((224, 224)),
        ]
    if len(x_augmentors) > 0:
        ds = AugmentImageComponent(ds, x_augmentors, copy=True)
    ds = AugmentImageComponents(ds, xy_augmentors, copy=False)
    ds = BatchData(ds, args.batch_size // args.nr_gpu, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def eval_on_camvid(ds):
    model = AnytimeFCDensenet(args)
    pred_config = PredictConfig(
        model=model,
        session_init=SaverRestore(args.load),
        input_names=['input', 'label'],
        output_names=['layer090.0.pred/confusion_matrix/SparseTensorDenseAdd:0']
        #, 'eval_mask:0', 'label']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    mean_iou = MeanIoUFromConfusionMatrix()
    mean_iou._before_inference()
    for o in pred.get_result():
        mean_iou._datapoint([o[0]])
    ret = mean_iou._after_inference()
    print ret


def get_config(ds_trian, ds_val, model_cls):
    # prepare dataset
    steps_per_epoch = ds_train.size() // args.nr_gpu

    model=model_cls(args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, keep_freq=12),
            InferenceRunner(ds_val,
                            [ScalarStats('cost')] + classification_cbs),
            #ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate')
        ] + loss_select_cbs,
        model=model,
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset choice 
    parser.add_argument('--ds_name', help='name of dataset',
                        type=str, 
                        choices=['camvid'])
    # other common args
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=3)
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
    parser.add_argument('--model_dir', help='model_dir position',
                        type=str, default=None)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--do_validation', help='Whether use validation set. Default not',
                        default=False, action='store_true')
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--is_toy', help='Whether to have data size of only 1024',
                        default=False, action='store_true')
    parser.add_argument('--is_test', help='Whehter use train-val or trainval-test',
                        default=False, action='store_true')
    parser.add_argument('--eval',  help='whether do evaluation only',
                        default=False, action='store_true')
    anytime_network.parser_add_fcdense_arguments(parser)
    model_cls = FCDensenet
    args = parser.parse_args()

    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()
    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))
    fs.set_dataset_path(args.data_dir)

    ## Set dataset-network specific assert/info
    #
    # Make sure the input images have H/W that are divisible by
    # 2**n_pools; see tensorpack/network_models/anytime_network.py
    if args.ds_name == 'camvid':
        args.num_classes = 11
        # the last weight is for void
        args.class_weight = dataset.Camvid.class_weight[:-1]
        INPUT_SIZE = None
        get_data = get_camvid_data
        if not args.is_test:
            ds_train = get_data('train') #trainval
            ds_val = get_data('val') #test
        else:
            ds_train = get_data('train')
            ds_val = get_data('test')

        if args.eval:
            eval_on_camvid(ds_val)
            sys.exit()

        #lr_schedule = \
        #    [(1, 0.001), (200, 1e-3 * 0.37), (400, 1e-3 * 0.37**2), (600, 1e-3 * 0.37**3)]
        max_epoch = 1000

    
    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
