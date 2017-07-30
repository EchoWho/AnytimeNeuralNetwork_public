import numpy as np
import os
from tensorpack import *



class RawILSVRC12(DataFlow):
    def __init__(self, split='train'):
        meta = dataset.ILSVRCMeta('/data2/ILSVRC2012/caffe_meta')
        self.imglist = meta.get_image_list('train')
# we apply a global shuffling here because later we'll only use local shuffling
        np.random.shuffle(self.imglist)
        self.dir = os.path.join('/data2/ILSVRC2012/raw', split)

    def get_data(self):
        for fname, label in self.imglist:
            fname = os.path.join(self.dir, fname)
            with open(fname, 'rb') as f:
                jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]

    def size(self):
        return len(self.imglist)


split='train'
ds0 = RawILSVRC12(split)
ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
dftools.dump_dataflow_to_lmdb(ds1, '/data2/ILSVRC2012/ilsvrc2012_{}.lmdb'.format(split))
