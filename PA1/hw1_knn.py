from __future__ import division, print_function

from typing import List

import numpy as np
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.train_features = np.array(features)
        self.train_labels = np.array(labels)
        self.train_cnt = len(labels)

    # TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        p_count = len(features)
        y_ = [0 for i in range(p_count)]
        for k in range(p_count):
            k_labels = self.get_k_neighbors(features[k])
            cnt0 = k_labels.count(0)
            cnt1 = self.k - cnt0
            if cnt0 <= cnt1:  # TODO: <= or < ?
                y_[k] = 1
        return y_

    # TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        dists = np.zeros(self.train_cnt)
        for i in range(self.train_cnt):
            dists[i] = self.distance_function(point, self.train_features[i])
        k_idxs = dists.argsort()[:self.k]
        return list(np.array(self.train_labels)[k_idxs])


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
