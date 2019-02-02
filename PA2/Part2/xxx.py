#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/31/19 7:48 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : xxx.py
# @Software: PyCharm
import numpy as np
import PA2.Part2.bm_classify as sol

from PA2.Part2.data_loader import smile_dataset_clear, smile_dataset_blur, data_loader_mnist


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)


import time

datasets = [(smile_dataset_clear(), 'Clear smile data', 3)
    , (smile_dataset_blur(), 'Blur smile data', 3)
    , (data_loader_mnist(), 'MNIST', 10)]

for data, name, num_classes in datasets:
    print('%s: %d class classification' % (name, num_classes))
    X_train, X_test, y_train, y_test = data
    for gd_type in ["sgd", "gd"]:
        s = time.time()
        w, b = sol.multiclass_train(X_train, y_train, C=num_classes, gd_type=gd_type)
        print(gd_type + ' training time: %0.6f seconds' % (time.time() - s))
        train_preds = sol.multiclass_predict(X_train, w=w, b=b)
        preds = sol.multiclass_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f'
              % (accuracy_score(y_train, train_preds), accuracy_score(y_test, preds)))
    print()
