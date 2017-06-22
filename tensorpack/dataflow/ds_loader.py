import dataset_loaders #external library
import numpy as np
import sys

from .base import RNGDataFlow, DataFlow
from ..utils import logger 

class DatasetLoaderDataFlowLoadAll(RNGDataFlow):

    def __init__(self, datasetloader, shuffle=True):
        self.ds = datasetloader
        self.data = []
        for bi in range(self.ds.nbatches):
            sys.stdout.write("loading {} / {} \r".format(bi+1, self.ds.nbatches)),
            sys.stdout.flush()
            X,Y = self.ds.next()
            self.data.append([[X[k], Y[k]] for k in range(X.shape[0])])
        sys.stdout.write("\n")

        self.shuffle = shuffle
        assert self.ds.return_01c, self.ds.return_01c
        assert not self.ds.return_0_255, self.ds.return_0_255
        assert self.ds.return_list, self.ds.return_list
        assert not self.ds.fill_last_batch

    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs: 
            yield self.data[k]


class DatasetLoaderDataFlow(DataFlow):
    
    def __init__(self, datasetloader):
        self.ds = datasetloader
        assert self.ds.return_01c, self.ds.return_01c
        assert not self.ds.return_0_255, self.ds.return_0_255
        assert self.ds.return_list, self.ds.return_list

    def get_data(self):
        for _ in range(self.ds.nbatches):
            X, Y = self.ds.next()
            yield [ X, Y ] 

    def size(self):
        return self.ds.nbatches
