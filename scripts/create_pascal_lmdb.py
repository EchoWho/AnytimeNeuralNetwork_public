import numpy as np
import os
from tensorpack import *

## Data-set loaders from https://github.com/fvisin/dataset_loaders
from dataset_loaders import PascalVOCdataset

class PascalVOCDatasetLoaderDataFlow(DataFlow):
    def __init__(self, split, shuffle=False):
        self.loader = PascalVOCdataset(which_set=split,
                batch_size=1,
                seq_per_subset=0,
                seq_length=0,
                return_one_hot=False, # y is hw int32
                return_01c=True,   # x is nhwc 
                return_0_255=True, # x is hwc uint8
                return_list=True,
                use_threads=False,
                nthreads=1,
                shuffle_at_each_epoch=shuffle,
                fill_last_batch=True)
        self.loader.reset(shuffle=shuffle)

    def get_data(self):
        for i in range(self.loader.nbatches):
            dp = self.loader.next()
            yield [dp[0], dp[1]]

    def size(self):
        return self.loader.nbatches


lmdb_dir = "/home/hanzhang/ssd/pascal_voc_lmdb"

for split in ['train_extra', 'val']:
    shuffle = split !='val' 
    ds = PascalVOCDatasetLoaderDataFlow(split, shuffle)
    ds = PrefetchDataZMQ(ds, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds, 
        os.path.join(lmdb_dir, 'pascal_voc2012_{}.lmdb'.format(split)))
