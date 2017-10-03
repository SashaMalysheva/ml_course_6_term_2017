from matplotlib.mlab import find
import matplotlib.pyplot as pl
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class LinearSVM(object):
    def __init__(self, C=0.1):
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        K = X * y[:, np.newaxis]
        K = np.dot(K, K.T)
        N = len(X)

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(cvxopt.matrix(K), cvxopt.matrix(-np.ones((N, 1))), cvxopt.matrix(np.vstack((np.eye(N), -np.eye(N)))),
                 cvxopt.matrix(np.vstack((np.ones((N, 1)) * self.C, np.zeros((N, 1))))),
                 cvxopt.matrix(y.reshape(1, N)), cvxopt.matrix(np.zeros(1)))

        self.alpha = np.array(sol['x']).reshape(N)

        self.support_ = [i for i in range(N) if self.alpha[i] > 1e-3]
        self.w = (X * (self.alpha * y)[:, np.newaxis]).sum(axis=0)
        for i in range(N):
            if 0 < self.alpha[i] < self.C:
                self.b = y[i] - np.dot(self.w, X[i])
                break

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class KernelSVM(object):
    def __init__(self, C, kernel=linear_kernel, sigma=1.0, degree=2):
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)
        self.support_ = sv

    def decision_function(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            if i % 10000 == 0:
                print(int(i/len(X)))
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


if __name__ == "__main__":
    def gen_lin_separable_data():
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(np.array([0, 2]), cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(np.array([2, 0]), cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def gen_non_lin_separable_data():
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal([-1, 2], cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal([4, -4], cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal([1, -1], cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal([-4, 4], cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def gen_lin_separable_overlap_data():
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(np.array([0, 2]), cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(np.array([2, 0]), cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def split_data(X1, y1, X2, y2):
        X_train = np.vstack((X1[:90], X2[:90]))
        y_train = np.hstack((y1[:90], y2[:90]))
        X_test = np.vstack((X1[90:], X2[90:]))
        y_test = np.hstack((y1[90:], y2[90:]))
        return X_train, y_train, X_test, y_test


    def visualize(clf, X, y):
        border = .5
        h = .02
        x_min, x_max = X[:, 0].min() - border, X[:, 0].max() + border
        y_min, y_max = X[:, 1].min() - border, X[:, 1].max() + border

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        mesh = np.c_[xx.ravel(), yy.ravel()]
        z_class = clf.predict(mesh).reshape(xx.shape)

        # Put the result into a color plot
        pl.figure(1, figsize=(8, 6))
        pl.pcolormesh(xx, yy, z_class, cmap=pl.cm.summer, alpha=0.3)

        # Plot hyperplane and margin
        z_dist = clf.decision_function(mesh).reshape(xx.shape)
        pl.contour(xx, yy, z_dist, [0.0], colors='black')
        pl.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

        # Plot also the training points
        y_pred = clf.predict(X)

        ind_support = clf.support_
        ind_correct = list(set(find(y == y_pred)) - set(ind_support))
        ind_incorrect = list(set(find(y != y_pred)) - set(ind_support))
        pl.scatter(X[ind_correct, 0], X[ind_correct, 1], c=y[ind_correct], cmap=pl.cm.summer, alpha=0.9)
        pl.scatter(X[ind_incorrect, 0], X[ind_incorrect, 1], c=y[ind_incorrect], cmap=pl.cm.summer, alpha=0.9,
                   marker='*',
                   s=50)
        pl.scatter(X[ind_support, 0], X[ind_support, 1], c=y[ind_support], cmap=pl.cm.summer, alpha=0.9, linewidths=1.8,
                   s=40)

        pl.xlim(xx.min(), xx.max())
        pl.ylim(yy.min(), yy.max())
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train, X_test, y_test = split_data(X1, y1, X2, y2)
        clf = LinearSVM()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print(("%d out of %d predictions correct" % (correct, len(y_predict))))
        visualize(clf, X_train, y_train)


    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train, X_test, y_test = split_data(X1, y1, X2, y2)
        clf = KernelSVM(None)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print(("%d out of %d predictions correct" % (correct, len(y_predict))))
        print('yyyy')

        visualize(clf, X_train, y_train)


    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train, X_test, y_test = split_data(X1, y1, X2, y2)
        clf = LinearSVM(C=0.1)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print(("%d out of %d predictions correct" % (correct, len(y_predict))))
        visualize(clf, X_train, y_train)

#test_soft()
test_non_linear()
#test_linear()

