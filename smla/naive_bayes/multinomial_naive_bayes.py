# encoding:utf-8
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class MultinomialNB(object):
    def __init__(self, alpha=1):
        """
        alpha ∈ [0, 1]
        """
        self.alpha = alpha
        self.class_prior = None
        self.feature_proba = None

    def _cal_class_prior(self, y):
        """
        shape: (1, n_classes)
        """
        class_count = np.sum(y, axis=0)

        return class_count / class_count.sum()

    def _cal_feature_proba(self, X, y):
        """
        log(P(x_i|y)) = log((N_yi + α)) - log(N_y + n * α)
        """
        # (n_classes, n_samples) x (n_samples, n_features) = (n_classes, n_features)
        feature_count = np.dot(y.T, X)
        feature_sum = feature_count.sum(axis=1, keepdims=True)
        feature_prob = np.log(feature_count + self.alpha) - np.log(feature_sum + X.shape[1] * self.alpha)

        return feature_prob

    def fit(self, X, y):
        y = np.reshape(y, [-1, 1])
        y = OneHotEncoder(sparse=False).fit_transform(y)

        self.class_prior = self._cal_class_prior(y)
        self.feature_proba = self._cal_feature_proba(X, y)

        return self

    def predict(self, X):
        """
        np.dot(X, self.feature_proba.T) = (n_samples, n_features) x (n_features, n_classes) = (n_samples, n_classes)
        """
        proba = np.log(self.class_prior) + np.dot(X, self.feature_proba.T)
        label = np.argmax(proba, axis=1)

        return label


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print classification_report(y_test, model.predict(X_test))
