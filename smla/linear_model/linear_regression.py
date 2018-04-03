# encoding:utf-8

import numpy as np


class LinearRegression(object):
    def __init__(self, reg_str=0.0):
        self.W = None
        self.error = None
        self.lam = reg_str  # regularization strength

    def _linear_hypothesis(self, W, X):
        """
        h = wX + b
        """
        return np.sum(W * X, axis=1, keepdims=True)

    def _cal_gradient(self, W, X, y):
        """
        g = (Σ[(h - y) * x_j] + lam * W ) / m
        """
        h = self._linear_hypothesis(W, X)
        g = (np.sum((h - y) * X, axis=0, keepdims=True) + self.lam * W) / X.shape[0]
        return g

    def _cal_error(self, W, X, y):
        """
        loss = Σ[(h - y)^2] / (2 * m) + (lam * W^2) / (2 * m)
        """
        h = self._linear_hypothesis(W, X)
        error = (np.sum((h - y) ** 2) + np.sum(self.lam * W ** 2)) / (2 * X.shape[0])
        return error

    def _concat_bias(self, X):
        bias = np.ones([X.shape[0], 1])
        X = np.concatenate([X, bias], axis=1)
        return X

    def fit(self, X, y, epoch=100, learning_rate=0.1, tolerance=1e-4):
        n_samples, n_features = X.shape

        # consider x_0 = 1 which is the feature val of the bias
        X = self._concat_bias(X)
        y = np.reshape(y, [-1, 1])

        # init weights
        self.W = np.zeros([1, n_features + 1])

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
        return self._linear_hypothesis(self.W, X)


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    import matplotlib.pyplot as plt

    X, y = load_boston(True)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    plt.figure()
    plt.plot(range(0, y_test.shape[0]), y_test, 'g-', label='Real price')
    plt.plot(range(0, y_test.shape[0]), pred, 'r--', label='Predicted price')
    plt.xlabel('samples')
    plt.ylabel('price')
    plt.title('Boston house-prices')
    plt.legend(loc='upper left')
    plt.show()
