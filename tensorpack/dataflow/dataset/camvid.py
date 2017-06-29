from ..base import RNGDataFlow
from ...utils import logger,fs
import os
import numpy as np

def load_data_from_npzs(fnames):
    if not isinstance(fnames, list):
        fnames = [fnames]
    Xs = []
    Ys = []
    for fname in fnames:
        d = np.load(fname)
        logger.info('Loading from {}'.format(fname))
        X, Y = (d['X'], d['Y'])
        Xs.append(X)
        Ys.append(Y)
    return np.stack(X), np.stack(Y)

class Camvid(RNGDataFlow):
    name = 'camvid'
    non_void_nclasses = 11
    _void_labels = [11]

    # optional arguments
    data_shape = (360, 480, 3)
    mean = [0.39068785, 0.40521392, 0.41434407]
    std = [0.29652068, 0.30514979, 0.30080369]

    _cmap = {
       0: (128, 128, 128),    # sky
       1: (128, 0, 0),        # building
       2: (192, 192, 128),    # column_pole
       3: (128, 64, 128),     # road
       4: (0, 0, 192),        # sidewalk
       5: (128, 128, 0),      # Tree
       6: (192, 128, 128),    # SignSymbol
       7: (64, 64, 128),      # Fence
       8: (64, 0, 128),       # Car
       9: (64, 64, 0),        # Pedestrian
       10: (0, 128, 192),     # Bicyclist
       11: (0, 0, 0)}         # Void

    _mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
           4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
           9: 'pedestrian', 10: 'byciclist', 11: 'void'}
    
    def __init__(self, which_set, shuffle=True, pixel_z_normalize=True, data_dir=None):
        """
        which_set : one of train, val, test, trainval
        shuffle:
        data_dir: <data_dir> should contain train.npz, val.npz, test.npz
        """
        self.shuffle = shuffle
        self.pixel_z_normalize = pixel_z_normalize

        if data_dir is None:
            data_dir = fs.get_dataset_path('camvid')
        assert os.path.exists(data_dir)
        for set_name in ['train', 'val', 'test']:
            assert os.path.exists(os.path.join(data_dir, '{}.npz'.format(set_name)))

        assert which_set in ['train', 'val', 'test', 'trainval'],which_set
        if which_set == 'train':
            load_fns = ['train']
        elif which_set == 'val':
            load_fns = ['val']
        elif which_set == 'test':
            load_fns = ['test']
        else: #if which_set == 'trainval':
            load_fns = ['train', 'val'] 
        load_fns = map(lambda fn : os.path.join(data_dir, '{}.npz'.format(fn)), load_fns)

        self.X, self.Y = load_data_from_npzs(load_fns)
        assert self.X.dtype == 'uint8'

        freq_fn = os.path.join(data_dir, 'frequency_stats.npz')
        freq_d = np.load(freq_fn)
        self.natural_freq = freq_d['natural_freq'] 
        self.median_freq = freq_d['median_freq']

        def normalize_freq(f):
            return f / float(np.sum(f)) * len(f)
        self.natural_freq = normalize_freq(self.natural_freq)
        self.median_freq = normalize_freq(self.median_freq)

        min_non_zero = np.min(self.median_freq[self.median_freq != 0])
        self.median_freq += (self.median_freq == 0) * min_non_zero * 0.5


    def get_data(self):
        idxs = np.arange(len(self.X))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            X = np.asarray(self.X[k], dtype=np.float32) / 255.0
            if self.pixel_z_normalize:
                X = (X - Camvid.mean) / Camvid.std
            yield [X, self.Y[k]]

    def size(self):
        return len(self.X)
