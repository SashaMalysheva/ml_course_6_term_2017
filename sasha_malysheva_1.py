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
    y_test = []
    distance = []
    ans = np.empty(4)
    X_test = np.atleast_2d(X_test)
    for row in X_test:
        for j in range(len(X_train)):
            distance.append((dist(row, X_train[j]), y_train[j]))
        distance.sort(key=lambda tup: tup[1])
        distance = distance[:int(k)]
        for x in distance:
            ans[int(x[0])] += 1
        y_test.append(np.argmax(ans))
    return y_test


def loocv(X_train, y_train, dist):
    answer = []
    for k in range(1, len(y_train)):
        y_pred = []
        for i in range(len(y_train)):
            y_pred.append(knn(np.delete(X_train, i, axis=0), np.delete(y_train, i), X_train[i, :], k, dist))
        y_pred = np.squeeze(np.asarray(y_pred))
        precision = np.sum(accuracy(y_train, y_pred)[0])
        answer.append((k, precision))
    answer.sort(key=lambda tup: tup[1])
    (k, precision) = answer[0]
    return k


def show_plt(a):
    plt.scatter(*zip(*a))
    plt.ylabel('Sort')
    plt.xlabel('Distance')
    plt.show()


def find_accuracy(X, dist, ratio):
    X_train, y_train, X_test, y_test = train_test_split(X[:, 1:], X[:, 0], ratio)
    k = loocv(X_train, y_train, dist)
    y_pred = knn(X_train, y_train, X_test, k, dist)
    return accuracy(y_test, y_pred)


def find_best_score(X, dist):
    answer = []
    for ratio in range(5, 90, 5):
        precision = np.sum(find_accuracy(X, dist, ratio/100)[0])
        answer.append((ratio, precision))
        pprint.pprint(ratio)
    pprint.pprint(answer)
    show_plt(answer)

X = download_csv('wine.csv')
find_best_score(X, manheten_dist)
find_best_score(X, euclidean_dist)

