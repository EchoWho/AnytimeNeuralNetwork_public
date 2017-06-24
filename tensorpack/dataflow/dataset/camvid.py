from ..ds_loader import DatasetLoaderDataFlow, DatasetLoaderDataFlowLoadAll
from ..base import ProxyDataFlow

from dataset_loaders import CamvidDataset

class Camvid(ProxyDataFlow):
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
    
    def __init__(self, which_set, shuffle=True, load_all=False):
        assert which_set in ['train', 'val', 'test', 'trainval'],which_set

        if load_all:
            raise NotImplementedError("Preload data processing in tfpack is not implemented yet")

        is_train = which_set in ['train', 'trainval']
        if is_train:
            augm_kwargs = {
                'crop_size' : (224, 224),
                'horizontal_flip' : 1,
                'channel_shift_range' : 0.04, 
            }
        else:
            augm_kwargs = {
                'crop_size' : (320, 480)
            }
        
        self.ds_loader = CamvidDataset(
                which_set=which_set,
                batch_size=3,
                seq_per_subset=0,
                seq_length=0,
                data_augm_kwargs=augm_kwargs,
                return_one_hot=False,
                return_01c=True,
                return_0_255=False,
                return_list=True,
                use_threads=False,
                fill_last_batch=False,
                shuffle_at_each_epoch=shuffle,
                remove_mean=True,
                divide_by_std=True)
        
        if load_all:
            self.ds = DatasetLoaderDataFlowLoadAll(self.ds_loader, shuffle)
        else:
            self.ds = DatasetLoaderDataFlow(self.ds_loader)
