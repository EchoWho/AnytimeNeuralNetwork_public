import numpy as np
import os
from tensorpack import *



class RawILSVRC12(DataFlow):
    def __init__(self, split):
        meta = dataset.ILSVRCMeta('/data2/ILSVRC2012/caffe_meta')
        self.imglist = meta.get_image_list(split)
# we apply a global shuffling here because later we'll only use local shuffling
        if split == 'train':
            np.random.shuffle(self.imglist)
        self.dir = os.path.join('/data2/ILSVRC2012/raw', split)
        self.name = split
        self.dir_structure = 'train'
        self.synset = meta.get_synset_1000()

    def get_data(self):
        add_label_to_fname = (self.name != 'train' and self.dir_structure != 'original')
        for fname, label in self.imglist:
            if add_label_to_fname:
                fname = os.path.join(self.dir, self.synset[label], fname)
            else:
                fname = os.path.join(self.dir, fname)
            with open(fname, 'rb') as f:
                jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]

    def size(self):
        return len(self.imglist)


split='val'
ds0 = RawILSVRC12(split)
ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
dftools.dump_dataflow_to_lmdb(ds1, '/data2/ILSVRC2012/ilsvrc2012_{}.lmdb'.format(split))
