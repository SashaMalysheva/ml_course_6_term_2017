import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def download_csv(name):
    X = np.genfromtxt(name, delimiter=',')
    return X


# Эта функция возвращает X_train, y_train, X_test, y_test
def train_test_split(X, y, ratio):
    dlen = len(X) * ratio
    X, y = shuffle(X, y, random_state=0)
    return X[:dlen, :], y[:dlen], X[dlen:, :], y[dlen:]


def euclidean_dist(u, x):
    d = 0.0
    for i in range(len(x) - 1):
        d += pow((float(x[i]) - float(u[i])), 2)
        d = np.math.sqrt(d)
    return d


def manheten_dist(u, x):
    d = 0.0
    for i in range(len(x) - 1):
        d += np.math.fabs(x[i] - u[i])
        d = np.math.sqrt(d)
    return d


def accuracy(y_test, y_pred):
    tp = np.zeros(4)
    fp = np.zeros(4)
    fn = np.zeros(4)
    precision = np.zeros(4)
    recall = np.zeros(4)
    for i in range(y_test.size - 1):
        if y_test[i] == y_pred[i]:
            tp[int(y_test[i])] += 1
        else:
            fp[int(y_pred[i])] += 1
            fn[int(y_test[i])] += 1

    for i in range(0, 3):
        if tp[i] + fp[i] != 0:
            precision[i] += tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] != 0:
            recall[i] += tp[i] / (tp[i] + fn[i])
    return precision, recall


def knn(X_train, y_train, X_test, k, dist):
    distance = []
    X_test = np.atleast_2d(X_test)
    for row in X_test:
        for j in range(len(X_train)):
            distance.append((dist(row, X_train[j]), y_train[j]))
    return distance


def loocv(X_train, y_train, dist):
    i = 3
    k = knn(np.delete(X_train, i, axis=0), np.delete(y_train, i, axis=0), X_train[i, :], len(X_train), dist)
    pprint.pprint(k)
    show_plt(k)


def show_plt(a):
    plt.scatter(*zip(*a))
    plt.ylabel('Ration')
    plt.xlabel('CV')
    plt.show()



X = download_csv('wine.csv')
X_train, y_train, X_test, y_test = train_test_split(X[:, 1:], X[:, 0], 0.5)
loocv(X_train, y_train, manheten_dist)

