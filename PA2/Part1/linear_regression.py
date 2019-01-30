"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    Xw_y = np.matmul(X, w) - y
    err = np.matmul(Xw_y.T, Xw_y) / y.shape[0]
    #####################################################
    return err


###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here #
    XtX_inv = np.linalg.inv(np.matmul(X.T, X))
    w = np.matmul(np.matmul(XtX_inv, X.T), y)
    #####################################################
    return w


###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    XtX = np.matmul(X.T, X)
    eigenValues, _ = np.linalg.eig(XtX)
    idx = np.abs(eigenValues).argsort()
    minimum_eigValue = eigenValues[idx[0]]
    eye_mat = np.eye(X.shape[1], dtype=float)
    while minimum_eigValue < 1e-5:
        XtX += 1e-1 * eye_mat
        eigenValues, _ = np.linalg.eig(XtX)
        idx = np.abs(eigenValues).argsort()
        minimum_eigValue = eigenValues[idx[0]]
    XtX_inv = np.linalg.inv(XtX)
    w = np.matmul(np.matmul(XtX_inv, X.T), y)
    #####################################################
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here #
    XtX = np.matmul(X.T, X)
    eye_mat = np.eye(X.shape[1], dtype=float)
    mat_inv = np.linalg.inv(XtX + lambd * eye_mat)
    w = np.matmul(np.matmul(mat_inv, X.T), y)
    #####################################################
    return w


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    mse_hist = []
    power = [i for i in range(-19, 20)]
    lambd_hist = []
    for p in power:
        if p < 0:
            lambd = 1 / (10 ** (-p))
        else:
            lambd = 10 ** p
        lambd_hist.append(lambd)
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        err = mean_square_error(w, Xval, yval)
        mse_hist.append(err)
    best_idx = np.argmin(mse_hist)
    bestlambda = lambd_hist[best_idx]
    #####################################################
    return bestlambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    X_list = []
    for i in range(1, power + 1):
        X_list.append(np.power(X, i))
    X = np.concatenate(X_list, axis=1)
    #####################################################		
    return X
