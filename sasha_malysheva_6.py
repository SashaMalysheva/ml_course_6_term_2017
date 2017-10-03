import numpy as np
import scipy.io
from sklearn.cross_validation import train_test_split


def sigmoid(M):
    e = np.math.e
    return 2 * ((1 + e ** M) ** (-1))


def sigmoid_prime(M):
    e = np.math.e
    return -np.math.log(2) * (2 ** (M + 1)) * ((1 + e ** M) ** (-2))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x ** 2


def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


def result_interpretation(y):
    return np.argmax(y)


class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        self.num_layers = len(layers)
        self.size = layers
        self.weights = []
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

    def forward(self, x):
        inputs = [x]
        for l in range(self.num_layers - 1):
            dot_value = np.dot(inputs[l], self.weights[l])
            activation = self.activation(dot_value)
            inputs.append(activation)
        return inputs

    def train(self, X, y, learning_rate=0.2, epochs=1000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            i = np.random.randint(X.shape[0] - 1)
            outputs = self.forward([X[i]])
            expected = vectorized_result(y[i])
            deltas = self.backward(outputs, expected)
            self.update_weights(outputs, deltas, learning_rate)

    def update_weights(self, outputs, deltas, learning_rate):
        for i in range(len(self.weights)):
            layer = np.atleast_2d(outputs[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    def backward(self, outputs, y):
        error = y.T - outputs[-1]
        deltas = [error * self.activation_prime(outputs[-1])]
        for l in range(self.num_layers - 2, 0, -1):
            deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(outputs[l]))
        deltas.reverse()
        return deltas

    def predict(self, x):
        answer = []
        for i in range(len(x)):
            a = np.concatenate((np.ones(1).T, np.array(x[i])))
            for l in range(0, len(self.weights)):
                a = self.activation(np.dot(a, self.weights[l]))
            answer.append(result_interpretation(a))
        return answer


def print_precision_recall(y_test, y_pred):
    tp = np.zeros(10)
    fp = np.zeros(10)
    fn = np.zeros(10)
    precision = np.zeros(10)
    recall = np.zeros(10)
    for i in range(y_test.size - 1):
        if y_test[i] == y_pred[i]:
            tp[int(y_test[i])] += 1
        else:
            fp[int(y_pred[i])] += 1
            fn[int(y_test[i])] += 1
    print('precision: ')
    print(tp / (tp + fp))
    print('recall: ')
    print(tp / (tp + fn))
    print('error: ')
    print(1 - sum(tp)/len(y_test))

dataset = scipy.io.loadmat('mnist-original.mat')
trainX, testX, trainY, testY = train_test_split(
dataset['data'].T / 255.0, dataset['label'].squeeze().astype("int0"), test_size=0.3)

print('With sigmoid:')
for j in range(7):
    print('Hidden level with size' + str(j))
    nn = NeuralNetwork([trainX.shape[1], j * 50 + 100, 10])
    nn.train(trainX, trainY)
    predY = nn.predict(testX)
    print_precision_recall(testY, predY)


