import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorpack import *


# the literal value are in rgb. cv2 will read in bgr
ilsvrc_mean = [0.485, 0.456, 0.406][::-1] 
ilsvrc_std = [0.229, 0.224, 0.225][::-1]

def get_distill_target_data(subset, options):
    distill_target_fn = os.path.join(options.data_dir, 'distill_targets', 
        '{}_distill_target_{}.bin'.format(options.ds_name, subset))
    ds = BinaryData(distill_target_fn, options.num_classes)
    return ds

def join_distill_and_shuffle(ds, subset, options, buffer_size=None):
    ds_distill = get_distill_target_data(subset, options)
    ds = JoinData([ds, ds_distill])
    if buffer_size is None:
        buffer_size = ds.size()
    ds = LocallyShuffleData(ds, buffer_size)
    return ds

def get_cifar_augmented_data(subset, options, do_multiprocess=True):
    isTrain = subset == 'train' and do_multiprocess
    use_distill = isTrain and options.alter_label
    shuffle = isTrain and not options.alter_label
    if options.num_classes == 10:
        ds = dataset.Cifar10(subset, shuffle=shuffle)
    elif options.num_classes == 100:
        ds = dataset.Cifar100(subset, shuffle=shuffle)
    else:
        raise ValueError('Number of classes must be set to 10(default) or 100 for CIFAR')
    logger.info('{} set has n_samples: {}'.format(subset, len(ds.data)))
    pp_mean = ds.get_per_pixel_mean()
    if use_distill:
        ds = join_distill_and_shuffle(ds, subset, options)
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
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    if do_multiprocess:
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_svhn_augmented_data(subset, options, do_multiprocess=True):
    isTrain = subset == 'train' and do_multiprocess
    use_distill = isTrain and options.alter_label
    shuffle = isTrain and not options.alter_label
    pp_mean = dataset.SVHNDigit.get_per_pixel_mean()
    if isTrain:
        d1 = dataset.SVHNDigit('train', shuffle=shuffle)
        d2 = dataset.SVHNDigit('extra', shuffle=shuffle)
        if use_distill:
            d1 = join_distill_and_shuffle(d1, 'train', options)
            d2 = join_distill_and_shuffle(d2, 'extra', options)
        ds = RandomMixData([d1, d2])
    else:
        ds = dataset.SVHNDigit(subset, shuffle=shuffle)

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
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    if do_multiprocess:
        ds = PrefetchData(ds, 5, 5)
    return ds


def get_ilsvrc_augmented_data(subset, options, do_multiprocess=True):
    isTrain = subset == 'train'
    lmdb_path = os.path.join(options.data_dir, 'lmdb2', 'ilsvrc2012_{}.lmdb'.format(subset))
    ds = LMDBData(lmdb_path, shuffle=False)
    if isTrain:
        ds = LocallyShuffleData(ds, 1024*64)  # This is 64G~80G in memory images
    ds = PrefetchData(ds, 1024*8, 1) # prefetch around 8 G
    ds = LMDBDataPoint(ds, deserialize=True)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0) # BGR uint8 data
    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            """
            crop 8%~100% of the original image
            See `Going Deeper with Convolutions` by Google.
            """
            def _augment(self, img, _):
                h, w = img.shape[:2]
                area = h * w
                for _ in range(10):
                    targetArea = self.rng.uniform(0.08, 1.0) * area
                    aspectR = self.rng.uniform(0.75, 1.333)
                    ww = int(np.sqrt(targetArea * aspectR))
                    hh = int(np.sqrt(targetArea / aspectR))
                    if self.rng.uniform() < 0.5:
                        ww, hh = hh, ww
                    if hh <= h and ww <= w:
                        x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                        y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                        out = img[y1:y1 + hh, x1:x1 + ww]
                        out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_CUBIC)
                        return out
                out = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                return out

        augmentors = [
            Resize(),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256),
            imgaug.CenterCrop((224, 224)),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    if do_multiprocess:
        ds = PrefetchDataZMQ(ds, min(24, multiprocessing.cpu_count()))
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    return ds
