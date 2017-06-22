from ..ds_loader import DatasetLoaderDataFlow, DatasetLoaderDataFlowLoadAll
from ..base import ProxyDataFlow

from dataset_loaders import CamvidDataset

class Camvid(ProxyDataFlow):

    def __init__(self, which_set, shuffle=True, load_all=False):
        assert which_set in ['train', 'val', 'test', 'trainval'],which_set
        
        self.ds_loader = CamvidDataset(
                which_set=which_set,
                batch_size=1,
                seq_per_subset=0,
                seq_length=0,
                data_augm_kwargs={},
                return_one_hot=False,
                return_01c=True,
                return_0_255=False,
                return_list=True,
                use_threads=False,
                fill_last_batch=False,
                shuffle_at_each_epoch=shuffle)
        
        if load_all:
            self.ds = DatasetLoaderDataFlowLoadAll(self.ds_loader, shuffle)
        else:
            self.ds = DatasetLoaderDataFlow(self.ds_loader)

