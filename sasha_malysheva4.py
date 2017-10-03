import random
import re
import numpy as np
import math
from collections import Counter

from functools import reduce


def readfile(name):
    f = open(name, 'r')
    return f.read()


def get_dictionary(X):
    s = set()
    for msg in X:
        if msg != []:
            for word in msg[1:]:
                letters_only = re.sub("[^a-zA-Z]", " ", word)
                lower_case = letters_only.lower()
                from nltk.corpus import stopwords
                if not lower_case in stopwords.words("english"):
                    s.add(lower_case)
    return list(s)


def vectorize(X, y, dictionary):
    m = np.zeros((len(y), len(dictionary)))
    for i in range(len(y)):
        wordCount = Counter(X[i])
        for j in range(len(dictionary)):
            m[i, j] = wordCount[dictionary[j]]
    return m


def split_data(text):
    text = text.splitlines()
    X = []
    y = np.zeros(len(text))
    for i in range(len(text)):
        msg = text[i].split()
        X.append(msg)
        if msg != [] and msg[0] == 'ham':
            y[i] = 1
    return X, y


class NaiveBayes:
    def __init__(self, dictionary, alpha, answer=None):
        if answer is None:
            answer = [[]]
        self.alpha = alpha
        self.answer = answer
        self.dictionary = dictionary

    def fit(self, X, y):
        cs = np.unique(y)
        answer = []
        # apriori_probability, frequency_of_words, sum_word, len(self.dictionary), sum_of_words_without_counting
        m = vectorize(X, y, self.dictionary)
        for c in cs:
            answer.append([])
            apriori_probability = float(c * sum(y) + (1 - c) * (len(y) - sum(y))) / len(y)
            frequency_of_words_values = np.zeros(len(self.dictionary))
            sum_word = 0
            sum_word_without_counting = np.zeros(len(self.dictionary))
            for i in range(len(y)):
                if y[i] == c:
                    sum_word += sum(m[i, :])
                    frequency_of_words_values += m[i, :]
                    for j in range(m.shape[1]):
                        sum_word_without_counting[j] += 1 if m[i,j] > 0 else 0
            answer[int(c)].append(apriori_probability)
            answer[int(c)].append(dict(zip(self.dictionary, frequency_of_words_values)))
            answer[int(c)].append(sum_word)
            answer[int(c)].append(len(self.dictionary))
            answer[int(c)].append(dict(zip(self.dictionary,sum_word_without_counting)))
        self.answer = answer

    def teta_wc(self, w, c):
        x = self.alpha + self.answer[c][1][w]
        y = self.alpha * self.answer[c][3] + self.answer[c][2]
        return math.log(x / y)

    def teta_wc1(self, w, c):
        x = self.alpha + self.answer[c][4][w]
        y = 2 * self.alpha + self.answer[c][2]
        return math.log(x / y)

    def predict(self, X):
        y = np.zeros(len(X))
        for msg_num in range(len(X)):
            probability = np.zeros(len(self.answer))
            for c in range(len(self.answer)):
                probability[c] += math.log(self.answer[c][0] * 100) - math.log(100)
                msg = X[msg_num]
                for j in range(1, len(msg)):
                    letters_only = re.sub("[^a-zA-Z]", " ", msg[j])
                    word = letters_only.lower()
                    probability[c] += self.teta_wc(word, c)
            y[msg_num] = np.argmax(probability)
        return y

    def predict1(self, X):
        y = np.zeros(len(X))
        for msg_num in range(len(X)):
            probability = np.zeros(len(self.answer))
            for c in range(len(self.answer)):
                probability[c] += math.log(self.answer[c][0] * 100) - math.log(100)
                msg = ', '.join([str(x) for x in list(X[msg_num])])
                letters_only = re.sub("[^a-zA-Z]", " ", msg)
                msg_words = set(letters_only.lower())
                for word in self.dictionary:
                    if word in msg_words:
                        probability[c] += self.teta_wc1(word, c)
                    else:
                        probability[c] += 1 - self.teta_wc1(word, c)
            y[msg_num] = np.argmax(probability)
        return y

    def score(self, X, y):
        ratio = 0.1
        dlen = int(len(y) * ratio)
        c = list(zip(X, y))
        random.shuffle(c)
        X, y = zip(*c)
        X_test = X[:dlen]
        X_train = X[dlen:]
        y_test = y[:dlen]
        y_train = y[dlen:]
        self.fit(X_train, y_train)
        if self.answer[0][0] * self.answer[1][0] == 0:
            print('Not enough samples')
            return 0
        y1 = self.predict1(X_test)
        return sum(np.ones(len(y1))[y1 == y_test]) / len(y1)


text = readfile('SMSSpamCollection')
X, y = split_data(text)
nb = NaiveBayes(get_dictionary(X), 1)
print(nb.score(X, y))
