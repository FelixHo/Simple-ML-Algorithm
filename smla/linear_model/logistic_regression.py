# encoding:utf-8
import numpy as np


class LogisticRegression(object):
    def __init__(self, reg_str=0.0):
        self.error = None
        self.W = None
        self.lam = reg_str  # regularization strength

    def _linear_hypothesis(self, W, X):
        """
        wX+b
        """
        return np.sum(W * X, axis=1, keepdims=True)

    def _sigmoid(self, X):
        """
        1/(1+e^-x)
        """
        return 1 / (1 + np.exp(-X))

    def _logistic_hypothesis(self, W, X):
        """
        1/(1 + e^-z)
        """
        return self._sigmoid(self._linear_hypothesis(W, X))

    def _cal_error(self, W, X, y):
        """
        Loss = Σ[log(1+e^z) - yz] / m + (lam * w^2 / 2*m)
        """
        z = self._linear_hypothesis(W, X)
        error = (np.sum(np.log(1 + np.exp(z)) - y * z) + np.sum(self.lam * W ** 2) / 2) / X.shape[0]
        return error

    def _cal_gradient(self, W, X, y):
        """
        g = (Σ(h-y)*x_j)/m + lam * w / m
        """
        h = self._logistic_hypothesis(W, X)
        return (np.sum((h - y) * X, axis=0, keepdims=True) + self.lam * W) / X.shape[0]

    def _concat_bias(self, X):
        bias = np.ones([X.shape[0], 1])
        X = np.concatenate([X, bias], axis=1)
        return X

    def fit(self, X, y, epoch=30, learning_rate=0.05, tolerance=1e-4):
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
        prob_true = self._logistic_hypothesis(self.W, X)
        prob = np.concatenate([1 - prob_true, prob_true], axis=1)
        label = np.argmax(prob, axis=1)
        return prob, label


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_curve
    import matplotlib.pyplot as plt

    X, y = load_breast_cancer(True)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LogisticRegression(reg_str=0.5)
    model.fit(X_train, y_train)
    pred_prob, pred_label = model.predict(X_test)
    print 'Accuracy: %.2f%%' % (np.average(pred_label == y_test) * 100)
    print 'F1-score: %.2f' % f1_score(y_true=y_test, y_pred=pred_label)

    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=pred_prob[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, 'g')
    plt.plot(np.linspace(0., 1., len(fpr)), np.linspace(0., 1., len(fpr)), '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
