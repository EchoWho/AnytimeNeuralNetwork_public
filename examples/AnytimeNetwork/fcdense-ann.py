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
from tensorpack.callbacks import JSONWriter, ScalarPrinter

from tensorpack.network_models import anytime_network
from tensorpack.network_models.anytime_network import AnytimeFCDensenet
import get_augmented_data


"""
"""
INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None
side = 224

def get_camvid_data(which_set, shuffle=True, slide_all=False):
    isTrain = (which_set == 'train' or which_set == 'trainval') and shuffle

    pixel_z_normalize = True 
    ds = dataset.Camvid(which_set, shuffle=shuffle, 
        pixel_z_normalize=pixel_z_normalize,
        is_label_one_hot=args.is_label_one_hot,
        slide_all=slide_all,
        slide_window_size=side,
        void_overlap=not isTrain)
    if isTrain:
        x_augmentors = [
            #imgaug.GaussianBlur(2)
        ]
        xy_augmentors = [
            #imgaug.RotationAndCropValid(7),
            #imgaug.RandomResize((0.8, 1.5), (0.8, 1.5), aspect_ratio_thres=0.0),
            #imgaug.RandomCrop((side, side)),
            imgaug.Flip(horiz=True),
        ]
    else:
        x_augmentors = []
        xy_augmentors = [
            #imgaug.RandomCrop((side, side)),
        ]
    if len(x_augmentors) > 0:
        ds = AugmentImageComponent(ds, x_augmentors, copy=True)
    ds = AugmentImageComponents(ds, xy_augmentors, copy=False)
    ds = BatchData(ds, args.batch_size // args.nr_gpu, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
    return ds

def get_pascal_voc_data(subset, do_multiprocess=True):
    return get_augmented_data.get_pascal_voc_augmented_data(subset, args, do_multiprocess)

def label_image_to_rgb(label_img, cmap):
    if len(label_img.shape) > 2:
        label_img = np.argmax(label_img, axis=-1)
    H, W = (label_img.shape[0], label_img.shape[1])
    return np.asarray([ cmap[y] for y in label_img.reshape([-1])], dtype=np.uint8).reshape([H,W,3])

def evaluate(subset, get_data, model_cls, meta_info):
    if logger.LOG_DIR is not None and args.display_period > 0:
        import matplotlib.pyplot as plt
        from scipy.misc import imresize
    args.batch_size = 1
    cmap = meta_info._cmap
    mean = meta_info.mean
    std = meta_info.std
    
    ds = get_data(subset, args, False)
    model = model_cls(args)

    l_outputs = [] 
    n_preds = np.sum(model.weights > 0)
    for i, weight in enumerate(model.weights):
        if weight > 0:
            l_outputs.extend(\
                ['layer{:03d}.0.pred/confusion_matrix/SparseTensorDenseAdd:0'.format(i),
                 'layer{:03d}.0.pred/pred_prob_img:0'.format(i)])
    
    pred_config = PredictConfig(
        model=model,
        session_init=SaverRestore(args.load),
        input_names=['input', 'label'],
        output_names=['input', 'label'] + l_outputs
    )
    pred = SimpleDatasetPredictor(pred_config, ds)

    l_total_confusion = [0] * n_preds
    for i, output in enumerate(pred.get_result()):
        img = np.asarray((output[0][0] * std + mean)*255, dtype=np.uint8)
        label = output[1][0]
        if len(label.shape) == 3: 
            mask = label.sum(axis=-1) < args.eval_threshold
        else:
            mask = label < args.num_classes
        label_img = label_image_to_rgb(label, cmap)  

        confs = output[2:][::2]
        preds = output[2:][1::2]

        for predi, perf in enumerate(zip(confs, preds)):
            conf, pred = perf
            l_total_confusion[predi] += conf

        if args.display_period > 0 and i % args.display_period == 0 \
                and logger.LOG_DIR is not None:
            save_dir = logger.LOG_DIR
            
            select_indices = [ int(np.round(n_preds * fraci / 4.0)) - 1 \
                            for fraci in range(1,5) ]
            preds_to_display = [ preds[idx] for idx in select_indices]
            fig, axarr = plt.subplots(1, 2+4, figsize=(1+6*6, 5))
            axarr[0].imshow(img)
            axarr[1].imshow(label_img) 
            for predi, pred in enumerate(preds_to_display):
                pred_img = pred[0].argmax(axis=-1)
                pred_img = label_image_to_rgb(pred_img, cmap)
                pred_img = imresize(pred_img, (img.shape[0], img.shape[1]))
                axarr[2+predi].imshow(pred_img) 

            plt.savefig(os.path.join(logger.LOG_DIR, 'img_{}.png'.format(i)),
                dpi=fig.dpi, bbox_inches='tight')

    #for each sample
    
    l_ret = []
    for i, total_confusion in enumerate(l_total_confusion):
        ret = dict()
        ret['subset'] = subset
        ret['confmat'] = total_confusion
        I = np.diag(total_confusion)
        n_true = np.sum(total_confusion, axis=1)
        n_pred = np.sum(total_confusion, axis=0)
        U = n_true + n_pred - I
        U += U==0
        IoUs = np.float32(I) / U
        mIoU = np.mean(IoUs)
        ret['IoUs'] = IoUs
        ret['mIoU'] = mIoU
        logger.info("ret info: {}".format(ret))

    if logger.LOG_DIR:
        npzfn = os.path.join(logger.LOG_DIR, 'evaluation.npz') 
        np.savez(npzfn, evalution_ret=ret)

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
                        choices=['camvid', 'pascal'])
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
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--is_test', help='Whehter use train-val or trainval-test',
                        default=False, action='store_true')
    parser.add_argument('--eval',  help='whether do evaluation only',
                        default=False, action='store_true')
    parser.add_argument('--finetune',  help='whether do fine tuning',
                        default=False, action='store_true')
    parser.add_argument('--display_period', help='Display at eval every # of image; 0 means no display',
                        default=0, type=int)
    anytime_network.parser_add_fcdense_arguments(parser)
    model_cls = AnytimeFCDensenet
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
        args.num_classes = dataset.Camvid.non_void_nclasses
        # the last weight is for void
        args.class_weight = dataset.Camvid.class_weight[:-1]
        args.optimizer = 'rmsprop'
        INPUT_SIZE = None
        get_data = get_camvid_data
        if args.eval:
            subset = 'test' if args.is_test else 'val'
            evaluate(subset, get_data, model_cls, dataset.Camvid)
            sys.exit()

        if not args.is_test:
            ds_train = get_data('train') #trainval
            ds_val = get_data('val') #test
        else:
            ds_train = get_data('train')
            ds_val = get_data('test')

        max_epoch = 750
        lr = args.init_lr
        lr_schedule = []
        for i in range(max_epoch):
            lr *= 0.995
            lr_schedule.append((i+1, lr))

    elif args.ds_name == 'pascal':
        args.num_classes = 22
        args.class_weight = np.ones(args.num_classes, dtype=np.float32)
        args.class_weight[0] = 1e-3
        INPUT_SIZE = None
        get_data = get_pascal_voc_data

        if args.eval:
            subset = 'val'
            evaluate(subset, get_data, model_cls, PascalVOC)
            sys.exit()

        ds_train = get_data('train_extra')
        ds_val = get_data('val')

        max_epoch = 40
        lr = args.init_lr
        lr_schedule = []
        for i in range(max_epoch):
            lr *= args.init_lr * (1.0 - i / np.float32(max_epoch))**0.9
            lr_schedule.append((i+1, lr))
    
    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
