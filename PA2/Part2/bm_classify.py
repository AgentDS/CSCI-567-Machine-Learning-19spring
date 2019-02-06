import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    # pre-processing for label data
    y = y * 2 - 1
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        for i in range(max_iterations):
            z = (np.matmul(X, w) + b) * y
            w += step_size * w_avg_gradient(X, y, z, loss)
            b += step_size * b_avg_gradient(X, y, z, loss)

    ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        for i in range(max_iterations):
            z = (np.matmul(X, w) + b) * y
            w += step_size * w_avg_gradient(X, y, z, loss)
            b += step_size * b_avg_gradient(X, y, z, loss)

    ############################################

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def w_avg_gradient(X, y, z, loss="perceptron"):
    N, D = X.shape
    if loss == "perceptron":
        sgn = (z <= 0).astype(float)
        all_gradient = np.matmul(X.T, sgn * y)
    elif loss == "logistic":
        all_gradient = np.matmul(X.T, sigmoid(-z) * y)
    else:
        raise "Loss Function is undefined."

    return all_gradient / N


def b_avg_gradient(X, y, z, loss="perceptron"):
    N, D = X.shape
    if loss == "perceptron":
        sgn = (z <= 0).astype(float)
        all_gradient = np.dot(sgn, y)
    elif loss == "logistic":
        all_gradient = np.dot(sigmoid(-z), y)
    else:
        raise "Loss Function is undefined."

    return all_gradient / N


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        z = np.matmul(X, w) + b
        preds = (z > 0).astype(int)
        ############################################
    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        z = sigmoid(np.matmul(X, w) + b)
        preds = (z > 0.5).astype(int)
        ############################################
    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    w_ = np.concatenate([w, b.reshape(C, 1)], axis=1)
    X_ = np.concatenate([X, np.ones((N, 1))], axis=1)

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        for i in range(max_iterations):
            n = np.random.choice(N)
            w_ -= step_size * multi_w_gradient(X_[n, :], y[n], C, w_, gd_type)
        w = w_[:, :D]
        b = w_[:, D]
    ############################################

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        # for i in range(max_iterations):
        #
        for i in range(max_iterations):
            w_ -= step_size * multi_w_gradient(X_, y, C, w_, gd_type)
        w = w_[:, :D]
        b = w_[:, D]
    ############################################

    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


# TODO
def multi_w_gradient(x, y, C, w, gd_type="sgd"):
    if gd_type == "sgd":
        one_hot = np.zeros(C)
        one_hot[y] = 1
        z = np.max(np.matmul(w, x))
        exps = np.exp(z - z.max())  # shape is (C,)
        sum_exp = np.sum(exps)
        tmp = exps / sum_exp - one_hot
        gradient = np.matmul(tmp.reshape(C, 1), x.reshape(1, x.shape[0]))
    elif gd_type == 'gd':
        N, D = x.shape
        one_hot = np.eye(C)[y]
        exps = np.exp(np.matmul(x, w.T))  # shape is (N,C)
        sum_exp = np.sum(exps, axis=1)  # shape is (N,)
        tmp = exps.T / sum_exp - one_hot.T  # shape is (C,N)
        gradient = np.matmul(tmp, x) / N
    else:
        raise "Type of Gradient Descent is undefined."
    return gradient


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.argmax(np.matmul(X, w.T) + b, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds
