#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ilsvrc.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import os
import tarfile
import cv2
import six
import numpy as np
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from ...utils import logger
from ...utils.loadcaffe import get_caffe_pb
from ...utils.fs import mkdir_p, download, get_dataset_path
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

import _ilsvrc_vgg_preprocess
import _ilsvrc_inception_preprocess

slim = tf.contrib.slim

__all__ = ['ILSVRCMeta', 'ILSVRC12', 'ILSVRC12TFRecord']

CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_integer('batch_size', 32,
#        """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_integer('image_size', 299,
#        """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 16,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 16,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('train_queue_capacity', 16,
                            """training queue capacity""")
# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
        """Size of the queue of preprocessed images. """
        """Default is ideal but try smaller values, e.g. """
        """4, 2 or 1, if host memory is constrained. See """
        """comments in code for more details.""")


class ILSVRCMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    """

    def __init__(self, dir=None):
        if dir is None:
            dir = get_dataset_path('ilsvrc_metadata')
        self.dir = dir
        mkdir_p(self.dir)
        self.caffepb = get_caffe_pb()
        f = os.path.join(self.dir, 'synsets.txt')
        if not os.path.isfile(f):
            self._download_caffe_meta()

    def get_synset_words_1000(self):
        """
        Returns:
            dict: {cls_number: cls_name}
        """
        fname = os.path.join(self.dir, 'synset_words.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_synset_1000(self):
        """
        Returns:
            dict: {cls_number: synset_id}
        """
        fname = os.path.join(self.dir, 'synsets.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def _download_caffe_meta(self):
        fpath = download(CAFFE_ILSVRC12_URL, self.dir)
        tarfile.open(fpath, 'r:gz').extractall(self.dir)

    def get_image_list(self, name):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname)
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                ret.append((name, int(cls)))
        assert len(ret)
        return ret

    def get_per_pixel_mean(self, size=None):
        """
        Args:
            size (tuple): image size in (h, w). Defaults to (256, 256).
        Returns:
            np.ndarray: per-pixel mean of shape (h, w, 3 (BGR)) in range [0, 255].
        """
        obj = self.caffepb.BlobProto()

        mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
        with open(mean_file, 'rb') as f:
            obj.ParseFromString(f.read())
        arr = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
        arr = np.transpose(arr, [1, 2, 0])
        if size is not None:
            arr = cv2.resize(arr, size[::-1])
        return arr


class ILSVRC12(RNGDataFlow):
    """
    Produces uint8 ILSVRC12 images of shape [h, w, 3(BGR)], and a label between [0, 999],
    and optionally a bounding box of [xmin, ymin, xmax, ymax].
    """
    def __init__(self, dir, name, meta_dir=None, shuffle=None,
                 dir_structure='original', include_bb=False):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``, where the
                original ``ILSVRC12_img_{name}.tar`` gets decompressed.
            name (str): 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): The dir structure of 'val' and 'test' directory.
                If is 'original', it expects the original decompressed
                directory, which only has list of image files (as below).
                If set to 'train', it expects the same two-level
                directory structure simlar to 'train/'.
            include_bb (bool): Include the bounding box. Maybe useful in training.

        Examples:

        When `dir_structure=='original'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                ILSVRC2012_val_00000001.JPEG
                ...
              test/
                ILSVRC2012_test_00000001.JPEG
                ...

        With ILSVRC12_img_*.tar, you can use the following
        command to build the above structure:

        .. code-block:: none

            mkdir val && tar xvf ILSVRC12_img_val.tar -C val
            mkdir test && tar xvf ILSVRC12_img_test.tar -C test
            mkdir train && tar xvf ILSVRC12_img_train.tar -C train && cd train
            find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        meta = ILSVRCMeta(meta_dir)
        self.imglist = meta.get_image_list(name)
        self.dir_structure = dir_structure
        self.synset = meta.get_synset_1000()

        if include_bb:
            bbdir = os.path.join(dir, 'bbox') if not \
                isinstance(include_bb, six.string_types) else include_bb
            assert name == 'train', 'Bounding box only available for training'
            self.bblist = ILSVRC12.get_training_bbox(bbdir, self.imglist)
        self.include_bb = include_bb

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        add_label_to_fname = (self.name != 'train' and self.dir_structure != 'original')
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            if add_label_to_fname:
                fname = os.path.join(self.full_dir, self.synset[label], fname)
            else:
                fname = os.path.join(self.full_dir, fname)
            im = cv2.imread(fname.strip(), cv2.IMREAD_COLOR)
            assert im is not None, fname
            if im.ndim == 2:
                im = np.expand_dims(im, 2).repeat(3, 2)
            if self.include_bb:
                bb = self.bblist[k]
                if bb is None:
                    bb = [0, 0, im.shape[1] - 1, im.shape[0] - 1]
                yield [im, label, bb]
            else:
                yield [im, label]

    @staticmethod
    def get_training_bbox(bbox_dir, imglist):
        ret = []

        def parse_bbox(fname):
            root = ET.parse(fname).getroot()
            size = root.find('size').getchildren()
            size = map(int, [size[0].text, size[1].text])

            box = root.find('object').find('bndbox').getchildren()
            box = map(lambda x: float(x.text), box)
            # box[0] /= size[0]
            # box[1] /= size[1]
            # box[2] /= size[0]
            # box[3] /= size[1]
            return np.asarray(box, dtype='float32')

        with timed_operation('Loading Bounding Boxes ...'):
            cnt = 0
            import tqdm
            for k in tqdm.trange(len(imglist)):
                fname = imglist[k][0]
                fname = fname[:-4] + 'xml'
                fname = os.path.join(bbox_dir, fname)
                try:
                    ret.append(parse_bbox(fname))
                    cnt += 1
                except KeyboardInterrupt:
                    raise
                except:
                    ret.append(None)
            logger.info("{}/{} images have bounding box.".format(cnt, len(imglist)))
        return ret



class ILSVRC12TFRecord(RNGDataFlow):
    def __init__(self, tfrecord_dir, subset, batch_size, height=224, width=224, preprocess='vgg'):
        assert preprocess in ['vgg', 'inceptin'], preprocess
        if subset[:4] == 'toy_':
            self.subset = subset[4:]
            self.is_toy = True
        else:
            self.subset = subset
            self.is_toy = False
        self.tfrecord_dir = tfrecord_dir
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.preprocess = preprocess
        # construct graph 
        self._get_data()

    def size(self):
        n_samples = 0
        if self.is_toy:
            n_samples = 1024
        elif self.subset == 'train':
            n_samples = 1281167
        elif self.subset == 'validation':
            n_samples = 50000
        n_batch = n_samples // self.batch_size
        remainder = n_samples % self.batch_size
        #if remainder == 0:
        #    return n_batch
        #return n_batch + int(self.subset == 'validation')
        return n_batch


    def data_files(self):
        tf_record_pattern = os.path.join(self.tfrecord_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s at %s' % (self.subset,
                                                           self.tfrecord_dir))

            exit(-1)
        return data_files
 
    def _get_data(self):
        """
            construct preprocessing graph on CPU. (copied from tf-slim)
        """
        with tf.device('/cpu:0'):
            with tf.name_scope('get_data'):
                data_files = self.data_files()
                train = self.subset == 'train'
                if train:
                    train_queue_capacity = FLAGS.train_queue_capacity
                    filename_queue = tf.train.string_input_producer(data_files,
                                                                    shuffle=True,
                                                                    capacity=train_queue_capacity)
                else:
                    filename_queue = tf.train.string_input_producer(data_files,
                                                                    shuffle=False,
                                                                    capacity=1)
                num_preprocess_threads = FLAGS.num_preprocess_threads
                if num_preprocess_threads % 4:
                    raise ValueError('Please make num_preprocess_threads a multiple '
                                     'of 4 (%d % 4 != 0).', num_preprocess_threads)

                num_readers = FLAGS.num_readers
                if num_readers < 1:
                    raise ValueError('Please make num_readers at least 1')

                # Approximate number of examples per shard.
                examples_per_shard = 1024
                # Size the random shuffle queue to balance between good global
                # mixing (more examples) and memory use (fewer examples).
                # 1 image uses 299*299*3*4 bytes = 1MB
                # The default input_queue_memory_factor is 16 implying a shuffling queue
                # size: examples_per_shard * 16 * 1MB = 17.6GB
                min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
                if train:
                     examples_queue = tf.RandomShuffleQueue(
                        capacity=min_queue_examples + 3 * self.batch_size,
                        min_after_dequeue=min_queue_examples,
                        dtypes=[tf.string])
                else:
                    examples_queue = tf.FIFOQueue(
                        capacity=examples_per_shard + 3 * self.batch_size,
                        dtypes=[tf.string])

                # Create multiple readers to populate the queue of examples.
                if num_readers > 1:
                  enqueue_ops = []
                  for _ in range(num_readers):
                    reader = tf.TFRecordReader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                  tf.train.queue_runner.add_queue_runner(
                      tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                  example_serialized = examples_queue.dequeue()
                else:
                  reader = tf.TFRecordReader()
                  _, example_serialized = reader.read(filename_queue)

                inputs = []
                for thread_id in range(num_preprocess_threads):
                    image_buffer, label_index, bbox, _ = ILSVRC12TFRecord.parse_example_proto(
                        example_serialized)
                    image = ILSVRC12TFRecord.decode_jpeg(image_buffer)
                    if self.preprocess == 'vgg':
                        image = _ilsvrc_vgg_preprocess.preprocess_image(image,
                            self.height, self.width, train)
                    elif self.preprocess == 'inception':
                        image = _ilsvrc_inception_preprocess.preprocess(image,
                            self.height, self.width, train, bbox)

                    #image_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
                    #image_std_inv = 1.0 / np.array([0.229, 0.224, 0.225], dtype='float32')
                    #image = tf.subtract(image, image_mean)
                    #image = tf.multiply(image, image_std_inv)
                    
                    inputs.append([image, label_index])

                images, labels = tf.train.batch(
                    inputs,
                    batch_size=self.batch_size,
                    num_threads=num_preprocess_threads, 
                    capacity=5 * self.batch_size,
                    allow_smaller_final_batch=not train)
                
                batch_queue = slim.prefetch_queue.prefetch_queue(\
                    [images, labels], capacity=8)

                images, labels = batch_queue.dequeue()
                
                images = tf.cast(images, tf.float32)
                images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, 3])
                labels = tf.reshape(labels, [self.batch_size])
                
                self.im = images
                self.label = labels

    def get_data(self):
        for _ in range(self.size()):
            # it's better to run a whole bunch of images together and in parallel
            sess = tf.get_default_session()
            if sess is not None:
                im, label = sess.run([self.im, self.label])
            else:
                raise Exception("Expect there is a default session")

            # label -1 because imagenet default labels are 1 based. (0 is background)
            offset = 1
            yield [im, label - offset]

    @staticmethod
    def decode_jpeg(image_buffer, scope=None):
        """Decode a JPEG string into one 3-D float image Tensor.
            Args:
            image_buffer: scalar string Tensor.
            scope: Optional scope for op_scope.
            Returns:
            3-D float Tensor with values ranging from [0, 1).
        """
        with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
            # Decode the string as an RGB JPEG.
            # Note that the resulting image contains an unknown height and width
            # that is set dynamically by decode_jpeg. In other words, the height
            # and width of image is unknown at compile-time.
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            # The immediate result of decode is uint8. of [h,w,c] shape
            return image

    @staticmethod
    def parse_example_proto(example_serialized):
      """Parses an Example proto containing a training example of an image.
      The output of the build_image_data.py image preprocessing script is a dataset
      containing serialized Example protocol buffers. Each Example proto contains
      the following fields:
        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/synset: 'n03623198'
        image/class/text: 'knee pad'
        image/object/bbox/xmin: 0.1
        image/object/bbox/xmax: 0.9
        image/object/bbox/ymin: 0.2
        image/object/bbox/ymax: 0.6
        image/object/bbox/label: 615
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>
      Args:
        example_serialized: scalar Tensor tf.string containing a serialized
          Example protocol buffer.
      Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged as
          [ymin, xmin, ymax, xmax].
        text: Tensor tf.string containing the human-readable label.
      """
      # Dense features in Example proto.
      feature_map = {
          'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
          'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                  default_value=-1),
          'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value=''),
      }
      sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
      # Sparse features in Example proto.
      feature_map.update(
          {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                       'image/object/bbox/ymin',
                                       'image/object/bbox/xmax',
                                       'image/object/bbox/ymax']})

      features = tf.parse_single_example(example_serialized, feature_map)
      label = tf.cast(features['image/class/label'], dtype=tf.int32)

      xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
      ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
      xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
      ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

      # Note that we impose an ordering of (y, x) just to make life difficult.
      bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

      # Force the variable number of bounding boxes into the shape
      # [1, num_boxes, coords].
      bbox = tf.expand_dims(bbox, 0)
      bbox = tf.transpose(bbox, [0, 2, 1])

      return features['image/encoded'], label, bbox, features['image/class/text']


def test0():
    meta = ILSVRCMeta()
    # print(meta.get_synset_words_1000())

    ds = ILSVRC12('/home/wyx/data/fake_ilsvrc/', 'train', include_bb=True,
                  shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break

if __name__ == '__main__':
    test0()
