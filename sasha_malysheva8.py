from pylab import *
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize, Imputer, scale


def cost_function(X, y, theta):
    m = len(y)
    return np.sum((X.dot(theta) - y) ** 2) / 2 / m


def download_csv(name):
    X = np.genfromtxt(name, delimiter=',')
    return X


# Эта функция возвращает X_train, y_train, X_test, y_test
def train_test_split(X, y, ratio=0.2):
    dlen = len(X) * ratio
    X, y = shuffle(X, y, random_state=0)
    return X[:dlen, :], y[:dlen], X[dlen:, :], y[dlen:]


class NormalLR:
    def __init__(self, regularization=1):
        self.regularization=regularization

    def fit(self, X, y):
        E = np.zeros(X.T.dot(X).shape)
        np.fill_diagonal(E, self.regularization)
        XtX_xT = np.linalg.inv(X.T.dot(X) + E).dot(X.T)
        weights = XtX_xT.dot(y)
        self.weights = np.array(np.asmatrix(weights)).flatten()

    def predict(self, X):
        y = X.dot(self.weights)
        return np.array(y).flatten()


class GradientLR(NormalLR):
    def __init__(self, *, alpha=0.01, n_epoch=100, regularization=0):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        self.regularization = regularization
        self.alpha = alpha
        self.threshold = alpha / 100
        self.n_epoch = n_epoch

    def fit(self, X, y):
        weights = np.ones(X.shape[1])
        for iteration in range(self.n_epoch):
            hypothesis = X.dot(weights)
            loss = hypothesis - y + self.regularization * np.sqrt(weights.dot(weights))
            gradient = X.T.dot(loss) / len(y)
            weights -= self.alpha * gradient
        self.weights = np.array(weights).flatten()
        return self


def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


def sample(size, *, weights):
    X = np.ones((size, 2))
    X[:, 1] = np.random.gamma(4., 2., size)
    y = X.dot(np.asarray(weights))
    y += np.random.normal(0, 1, size)
    return X[:, 1:], y


def chek_on_sampels():
    gr = GradientLR(regularization=1)
    lr = NormalLR(regularization=1)
    answerlr = []
    answergr = []
    for i in range(3, 20):
        X, y_true = sample(size=2 ** i, weights=[24., 42.])
        lr.fit(X, y_true)
        gr.fit(X, y_true)
        answerlr.append((2 ** i, mse(y_true, lr.predict(X))))
        answergr.append((2 ** i, mse(y_true, gr.predict(X))))
    print(answergr)
    plt.plot(*zip(*answergr), color="red")
    plt.plot(*zip(*answerlr), color="blue")
    plt.show()


def boston():
    X = download_csv('boston.csv')
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    imp.transform(X)
    X = scale(X, axis=0)
    X = normalize(X)
    X_train, y_train, X_test, y_test = train_test_split(X[:, 1:], X[:, 0])
    gr = GradientLR(regularization=1)
    lr = NormalLR(regularization=1)
    lr.fit(X_train, y_train)
    gr.fit(X_train, y_train)
    print(mse(y_test, lr.predict(X_test)))
    print(mse(y_test, gr.predict(X_test)))
    print(lr.weights)


chek_on_sampels()
boston()