import numpy as np
from typing import List
from hw1_knn import KNN


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    num_cases = np.sum(branches)
    num_cls = len(branches[0])
    num_branches = len(branches)
    p = [0 for i in range(num_branches)]
    sub_entropy = [0 for i in range(num_branches)]
    for i in range(num_branches):
        num_cases_branch = np.sum(branches[i])
        p[i] = num_cases_branch / num_cases
        for j in range(num_cls):
            if branches[i][j] > 0:
                p_j = branches[i][j] / num_cases_branch
                sub_entropy[i] -= p_j * np.log2(p_j)
    tmp = 0
    for i in range(num_branches):
        tmp += sub_entropy[i] * p[i]
    Gain = S - tmp
    return float(Gain)


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    raise NotImplementedError


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])
    # print('\t' + indent, 'feature number:', node.f_len)  # TODO: for debug

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        # print('\t' + indent, 'feature value:', node.feature_uniq_split)  # TODO: for debug
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t',
                       deep=deep + 1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


# TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    numerator = 2 * np.dot(real_labels, predicted_labels)
    denominator = np.sum(real_labels) + np.sum(predicted_labels)
    f1 = numerator / denominator
    return float(f1)


# TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return float(np.linalg.norm(np.array(point1) - np.array(point2)))


# TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return float(np.dot(point1, point2))


# TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    diff = np.array(point1) - np.array(point2)
    return float(-np.exp(-0.5 * np.dot(diff, diff)))


# TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    numerator = np.dot(point1, point2)
    denominator = np.linalg.norm(point1) * np.linalg.norm(point2)
    return float(1 - numerator / denominator)


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model

    dist_names = ['euclidean', 'gaussian', 'inner_prod', 'cosine_dist']
    best_f1_iter_k = []
    best_dist_iter_k = []
    if len(ytrain) < 30:
        max_k = len(ytrain)
    else:
        max_k = 30
    for k in range(1, max_k, 2):
        f1_tmp = [0, 0, 0, 0]
        i = 0
        for dist in dist_names:
            func = distance_funcs[dist]
            model = KNN(k, func)
            model.train(Xtrain, ytrain)
            yval_ = model.predict(Xval)
            f1_tmp[i] = f1_score(yval, yval_)
            i = i + 1
        best_idx = np.argmax(f1_tmp)
        best_dist_iter_k.append(dist_names[best_idx])
        best_f1_iter_k.append(np.max(f1_tmp))
    best_idx = np.argmax(best_f1_iter_k)
    best_k = list(range(1, 30, 2))[best_idx]
    best_func = best_dist_iter_k[best_idx]
    best_model = KNN(best_k, distance_funcs[best_func])
    best_model.train(Xtrain, ytrain)

    return best_model, best_k, best_func


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model

    dist_names = ['euclidean', 'gaussian', 'inner_prod', 'cosine_dist']

    best_f1_iter_k = []
    best_dist_iter_k = []
    best_scale_iter_k = []

    norm_scaler = scaling_classes['normalize']()
    norm_Xtrain = norm_scaler(Xtrain)
    norm_Xval = norm_scaler(Xval)

    mm_scalar = scaling_classes['min_max_scale']()
    mm_Xtrain = mm_scalar(Xtrain)
    mm_Xval = mm_scalar(Xval)

    if len(ytrain) < 30:
        max_k = len(ytrain)
    else:
        max_k = 30
    for k in range(1, max_k, 2):
        f1_tmp = [0, 0, 0, 0]
        scaler_tmp = [None, None, None, None]
        i = 0
        for dist in dist_names:
            func = distance_funcs[dist]

            norm_model = KNN(k, func)
            norm_model.train(norm_Xtrain, ytrain)
            norm_yval_ = norm_model.predict(norm_Xval)
            norm_f1 = f1_score(yval, norm_yval_)

            mm_model = KNN(k, func)
            mm_model.train(mm_Xtrain, ytrain)
            mm_yval_ = mm_model.predict(mm_Xval)
            mm_f1 = f1_score(yval, mm_yval_)

            if norm_f1 > mm_f1:
                f1_tmp[i] = norm_f1
                scaler_tmp[i] = 'normalize'
            else:
                f1_tmp[i] = mm_f1
                scaler_tmp[i] = 'min_max_scale'
            i = i + 1
        best_idx = np.argmax(f1_tmp)
        best_dist_iter_k.append(dist_names[best_idx])
        best_f1_iter_k.append(np.max(f1_tmp))
        best_scale_iter_k.append(scaler_tmp[best_idx])

    best_idx = np.argmax(best_f1_iter_k)
    best_k = list(range(1, 30, 2))[best_idx]
    best_func = best_dist_iter_k[best_idx]
    best_scaler = best_scale_iter_k[best_idx]
    best_model = KNN(best_k, distance_funcs[best_func])
    if best_scaler == 'normalize':
        best_model.train(norm_Xtrain, ytrain)
    else:
        best_model.train(mm_Xtrain, ytrain)

    return best_model, best_k, best_func, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        cnt = len(features)
        normed = []
        for i in range(cnt):
            x = features[i]
            if any(x) is False:
                x_ = x
            else:
                x_ = list(np.array(x) / np.linalg.norm(x))
            normed.append(x_)
        return normed


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        self.max = None
        self.min = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        features = np.array(features)

        if self.max is None:
            self.max = features.max(axis=0)
        if self.min is None:
            self.min = features.min(axis=0)
        features = (features - self.min) / (self.max - self.min)
        diff = self.max - self.min
        zero_idx = np.where(diff == 0)[0].tolist()
        features[:, zero_idx] = 0
        return features.tolist()
