import numpy as np
import random

from scipy import sparse

from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler

class BaseMap():
    '''Intended as a spatial classifier template'''
    def __init__(self):
        #method and parameters
        pass
    def add_data(self):
        pass
    def fit(self):
        pass
    def classify(self):
        pass


#Test class syntax
obj = BaseMap()







