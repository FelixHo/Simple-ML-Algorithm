# encoding:utf-8

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class SoftmaxRegression(object):
    def __init__(self, reg_str=0.0):
        self.W = None
        self.error = None
        self.lam = reg_str  # regularization strength

    def _linear_hypothesis(self, W, X):
        """
        wX+b
        """
        return W.dot(X.T).T

    def _softmax(self, z):
        """
        e^zj/Σe^zk
        """
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def _softmax_hypothesis(self, W, X):
        """
        calculate probability
        """
        return self._softmax(self._linear_hypothesis(W, X))

    def _cal_error(self, W, X, y):
        """
        cross-entropy = [ΣΣ_k -y_k*log(h_k(x))]/m + (lam * w^2) / (2*m)
        """
        h = self._softmax_hypothesis(W, X)
        error = (np.sum(-y * np.log(h)) + np.sum(self.lam * W ** 2) / 2) / X.shape[0]
        return error

    def _cal_gradient(self, W, X, y):
        """
        g = Σ(h(x) - y)*x_j / m + lam * w / m
        """
        h = self._softmax_hypothesis(W, X)
        g = (np.dot(X.T, (h - y)).T + self.lam * W) / X.shape[0]
        return g

    def _concat_bias(self, X):
        bias = np.ones([X.shape[0], 1])
        X = np.concatenate([X, bias], axis=1)
        return X

    def fit(self, X, y, epoch=100, learning_rate=0.05, tolerance=1e-4):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # consider x_0 = 1 which is the feature val of the bias
        X = self._concat_bias(X)

        # [0,1,2] => [[0], [1], [2] ]
        y = np.reshape(y, [-1, 1])

        # [[0], [1], [2] ] => [[1,0,0], [0,1,0], [0,0,1]]
        y = OneHotEncoder(sparse=False).fit_transform(y)

        # init weights
        self.W = np.zeros([n_classes, n_features + 1])

        # mini-batch SGD
        mb = min(1000, n_samples / 10 or n_samples)

        for i in range(epoch):
            s = 0
            while s < n_samples:
                e = s + mb
                X_b = X[s:e]
                y_b = y[s:e]
                g = self._cal_gradient(self.W, X_b, y_b)
                self.W -= learning_rate * g
                self.error = self._cal_error(self.W, X, y)

                if self.error < tolerance:
                    return self
                s = e
        return self

    def predict(self, X_test):
        X = self._concat_bias(X_test)
        prob = self._softmax_hypothesis(self.W, X)
        label = np.argmax(prob, axis=1)
        return prob, label


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = SoftmaxRegression()
    model.fit(X_train, y_train)
    pred_prob, pred_label = model.predict(X_test)

    print 'Accuracy: %.2f%%' % (np.average(pred_label == y_test) * 100)
