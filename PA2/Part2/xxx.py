#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/31/19 7:48 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : xxx.py
# @Software: PyCharm
import numpy as np
import PA2.Part2.bm_classify as sol


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)


X_0 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X_b = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])  # x3 = x1x2
X_d = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 2]])  # x3 = x1^2 + X2^2
y = np.array([0, 1, 1, 0])

X = X_0
loss_type = "logistic"
w, b = sol.binary_train(X, y, loss=loss_type)
train_preds = sol.binary_predict(X, w, b, loss=loss_type)
print(loss_type + ' train acc: %f' % (accuracy_score(y, train_preds)))
print('w =', w)
print('b =', b)
print()
