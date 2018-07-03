import numpy as np
import os
from tensorpack import *

meta_dir = '/data2/ILSVRC2012/caffe_meta'
lmdb_dir = '/data2/ILSVRC2012/lmdb2'
raw_dir = '/data2/ILSVRC2012/raw'

dir_structure = 'train' #original 

class RawILSVRC12(DataFlow):
    def __init__(self, split, shuffle=True):
        meta = dataset.ILSVRCMeta(meta_dir)
        self.imglist = meta.get_image_list(split)
# we apply a global shuffling here because later we'll only use local shuffling
        if shuffle:
            np.random.shuffle(self.imglist)
        self.dir = os.path.join(raw_dir, split)
        self.name = split
        self.dir_structure = dir_structure
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


for split in ['train', 'val']:
    do_shuffle = split == 'train'
    ds0 = RawILSVRC12(split, shuffle=do_shuffle)
    np.savez(os.path.join(lmdb_dir, 'ilsvrc2012_{}_imglist.npz'.format(split)), imglist=ds0.imglist)
    ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds1, 
        os.path.join(lmdb_dir, 'ilsvrc2012_{}.lmdb'.format(split)))
