# encoding:utf-8
import numpy as np
import collections

class KNN(object):
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def _find_k_nearest(self, x):
        # brute-force search
        dist = np.linalg.norm(self.X - x, axis=1)

        return self.y[np.argsort(dist)[:self.k]]

    def fit(self, X, y):
        self.y = np.ravel(y)
        self.X = X

        return self

    def predict(self, X):
        label = []

        for each in X:
            neighbors = self._find_k_nearest(each)
            label.append(collections.Counter(neighbors).most_common()[0][0])

        return label

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNN(k=5)
    model.fit(X_train, y_train)
    label = model.predict(X_test)

    print classification_report(y_test, label)
    print confusion_matrix(y_test, label)


