import numpy as np
from Gradient_Descent import gradient_descent, stochastic_gradient_descent


class SVM:
    """

    """

    def __init__(self, learning_rate=0.0001, iterations=10000, batch_size=64, optim='GD'):
        """

        Args:
            learning_rate: the learning rate
            iterations: number of iterations we go through before training ends
            batch_size: the batch size, if SGD is used
            optim: the optimization method
        """
        self.lr = learning_rate
        self.iters = iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.optim = optim

    def fit(self, X, y):
        """

        Args:
            X: the datapoints of our dataset
            y: the classes of our dataset

        Returns: a trained model

        """
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        if self.optim == 'GD':
            self.weights, self.bias = gradient_descent(self.weights, self.bias, self.lr, X, y_,
                                                       self.iters)
        elif self.optim == 'SGD':
            self.weights, self.bias = stochastic_gradient_descent(self.weights, self.bias, self.lr, X, y_,

                                                                  self.batch_size, self.iters)
        else:
            print('no valid optimizer choosen, SGD will be used')
            self.weights, self.bias = stochastic_gradient_descent(self.weights, self.bias, self.lr, X, y_,

                                                                  self.batch_size, self.iters)

    def predict(self, X):
        """
        a simple method to predict the classes of a dataset
        Args:
            X: the datapoints in the dataset

        Returns: the predicted classes of the dataset

        """
        return np.sign(np.dot(X, self.weights) + self.bias)
