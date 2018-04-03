# encoding:utf-8
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class GaussianNB(object):
    def __init__(self):
        self.class_prior = None
        self.feature_mean = None
        self.feature_var = None

    def _cal_class_prior(self, y):
        """
        e.g.
        y = [[1,0]
             [0,1]]
        """
        class_count = np.sum(y, axis=0)
        return class_count / class_count.sum()

    def _cal_feature_mean_and_var(self, X, y):
        """
        e.g.
        y = [[1,0]
             [0,1]]
        """
        feature_mean = []
        feature_var = []
        epsilon = 1e-9 * np.var(X, axis=0).max()

        for i in range(y.shape[1]):
            class_feature = X[y[:, i] == 1]
            mean = np.mean(class_feature, axis=0)
            # prevent numerical errors. e.g when var=0
            var = np.var(class_feature, axis=0, ddof=1) + epsilon
            feature_mean.append(mean)
            feature_var.append(var)

        return feature_mean, feature_var

    def fit(self, X, y):
        y = np.reshape(y, [-1, 1])
        y = OneHotEncoder(sparse=False).fit_transform(y)
        self.class_prior = self._cal_class_prior(y)
        self.feature_mean, self.feature_var = self._cal_feature_mean_and_var(X, y)

        return self

    def predict(self, X):
        """
        log(gaussian) = - 0.5 * [log(2 * π * var) + (x - mean)^2 / var]
        """
        class_proba = []

        for i, cp in enumerate(self.class_prior):
            mean = self.feature_mean[i]
            var = self.feature_var[i]
            log_proba = np.log(cp) + np.sum(-0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var), axis=1, keepdims=True)
            class_proba.append(log_proba)

        proba = np.concatenate(class_proba, axis=1)
        label = np.argmax(proba, axis=1)

        return label


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)

    print classification_report(y_test, model.predict(X_test))
