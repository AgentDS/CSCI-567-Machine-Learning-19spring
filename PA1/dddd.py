#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/22/19 12:46 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : dddd.py
# @Software: PyCharm
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
import numpy as np
from PA1.hw1_knn import KNN
from PA1.utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance
from PA1.utils import f1_score, model_selection_without_normalization, model_selection_with_transformation
from PA1.data import data_processing

distance_funcs = {
    'euclidean': euclidean_distance,
    'gaussian': gaussian_kernel_distance,
    'inner_prod': inner_product_distance,
    'cosine_dist': cosine_sim_distance,
}

Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
model = KNN(5, inner_product_distance)
model.train(Xtrain, ytrain)
yval_ = model.predict(Xval)