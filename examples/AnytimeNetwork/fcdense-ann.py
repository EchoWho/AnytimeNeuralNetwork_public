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
from tensorpack.network_models.anytime_network import AnytimeFCDensenet

import matplotlib.pyplot as plt
import ipdb as pdb
from sklearn.metrics import confusion_matrix

"""
"""
INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None
side = 224

def get_camvid_data(which_set, shuffle=True, slide_all=False):
    isTrain = which_set == 'train' or which_set == 'trainval'

    pixel_z_normalize = True 
    ds = dataset.Camvid(which_set, shuffle=shuffle, 
        pixel_z_normalize=pixel_z_normalize,
        is_label_one_hot=args.is_label_one_hot,
        slide_all=slide_all,
        slide_window_size=side,
        void_overlap=not isTrain)
    if isTrain:
        if args.is_label_one_hot:
            x_augmentors = [
#                imgaug.GaussianBlur(2)
            ]
            xy_augmentors = [
#                imgaug.RotationAndCropValid(7),
#                imgaug.RandomResize((0.8, 1.5), (0.8, 1.5), aspect_ratio_thres=0.0),
                imgaug.RandomCrop((side, side)),
                imgaug.Flip(horiz=True),
            ]
        else:
            x_augmentors = []
            xy_augmentors = [ 
                imgaug.RandomCrop((side, side)),
                imgaug.Flip(horiz=True),
            ]
    else:
        x_augmentors = []
        xy_augmentors = [
            imgaug.RandomCrop((side, side)),
        ]
    if len(x_augmentors) > 0:
        ds = AugmentImageComponent(ds, x_augmentors, copy=True)
    ds = AugmentImageComponents(ds, xy_augmentors, copy=False)
    ds = BatchData(ds, args.batch_size // args.nr_gpu, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
    return ds

def label_image_to_rgb(label_img, cmap):
    H, W = (label_img.shape[0], label_img.shape[1])
    return np.asarray([ cmap[y] for y in label_img.reshape([-1])], dtype=np.uint8).reshape([H,W,3])

def eval_on_camvid(get_data):
    if args.is_test:
        which_set = 'test'
    else:
        which_set = 'val'
    ds = get_data(which_set, shuffle=False, slide_all=True)

    dscamvid = dataset.Camvid(which_set, shuffle=False, 
        is_label_one_hot=args.is_label_one_hot,
        slide_all=True,
        slide_window_size=side,
        void_overlap=True)

    model = AnytimeFCDensenet(args)
    pred_config = PredictConfig(
        model=model,
        session_init=SaverRestore(args.load),
        input_names=['input', 'label'],
        output_names=['layer090.0.pred/confusion_matrix/SparseTensorDenseAdd:0',
            'layer090.0.pred/pred_prob:0']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    l_output_list = []
    img_idx = -1
    plt.close('all')
    n_imgs=None
    l_imgs=[]
    total_confusion = None
    for i, o in enumerate(pred.get_result()):
        l_output_list.append(o)
        pred = o[1]
        if i % 2 == 0:
            l_imgs = []
            n_imgs = pred.shape[0] / side**2
            l_imgs.extend(pred.reshape((n_imgs, side, side, -1)))
        else:
            l_imgs.extend(pred.reshape((n_imgs,side,side,-1)))
            if (i+1) %2 ==0:
                img_idx += 1
                pred = dscamvid.stitch_sliding_images(l_imgs)
                pred_lbl = np.argmax(pred, axis=-1)
                image = dscamvid.X[img_idx]
                label = dscamvid.Y[img_idx]
                ds_cmap = dataset.Camvid._cmap
                pred_img = label_image_to_rgb(pred_lbl, ds_cmap)
                label_img = label_image_to_rgb(label, ds_cmap)

                #if img_idx % 20 == 0:
                #    fig, axarr = plt.subplots(1,3)
                #    axarr[0].imshow(image)
                #    axarr[1].imshow(label_img)
                #    axarr[2].imshow(pred_img)
                #    plt.show(block=False)

                mask = label.reshape([-1]) < args.num_classes
                cm = confusion_matrix(pred_lbl.reshape([-1])[mask], label.reshape([-1])[mask],
                    labels=np.arange(args.num_classes))
                if total_confusion is None:
                    total_confusion = cm
                else:
                    total_confusion += cm
        
                
    ret = dict()
    ret['confmat'] = total_confusion
    I = np.diag(total_confusion)
    n_true = np.sum(total_confusion, axis=1)
    n_pred = np.sum(total_confusion, axis=0)
    U = n_true + n_pred - I
    IoUs = np.float32(I) / U
    mIoU = np.mean(IoUs)
    ret['IoUs'] = IoUs
    ret['mIoU'] = mIoU
    pdb.set_trace()
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
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
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
    parser.add_argument('--finetune',  help='whether do fine tuning',
                        default=False, action='store_true')
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
        INPUT_SIZE = None
        get_data = get_camvid_data
        if args.eval:
            eval_on_camvid(get_data)
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

    
    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
