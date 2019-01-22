import numpy as np
from typing import List
from hw1_knn import KNN


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    raise NotImplementedError


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

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
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
    f1_dict = dict()
    k_dict = dict()
    train_cnt = len(ytrain)
    for dist in distance_funcs:
        f1_dict[dist] = 0.0
        k_dict[dist] = 0

    for dist in distance_funcs:
        func = distance_funcs[dist]
        for k in range(1, train_cnt + 1):
            model = KNN(k, func)
            model.train(Xtrain, ytrain)
            yval_ = model.predict(Xval)
            f1 = f1_score(yval, yval_)
            if f1 > f1_dict[dist]:
                f1_dict[dist] = f1
                k_dict[dist] = k
    dists = ['euclidean', 'gaussian', 'inner_prod', 'cosine_dist']
    best_idx = 0
    for i in range(1, 4):
        if f1_dict[dists[i]] > f1_dict[dists[best_idx]]:
            best_idx = i
    best_func = dists[best_idx]
    best_k = k_dict[best_func]
    best_model = KNN(best_k, distance_funcs[best_func])

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
    raise NotImplementedError


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
        raise NotImplementedError


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
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        raise NotImplementedError
