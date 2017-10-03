import pprint
import random
from itertools import izip
import numpy as np
import cv2


def read_image(path):
    img = cv2.imread(path)
    return img.reshape((img.shape[0] * img.shape[1], 3))


def euclidean_dist(x, y):
    return np.sqrt(np.sum((x-y)**2, axis=1))


def find_dist(X, centroids, distance_metric):
    dist = np.zeros([centroids.shape[0], X.shape[0]])
    for i in range(centroids.shape[0]):
        dist[i] = distance_metric(X, centroids[i])
    return dist


def closest_centroid(distance):
    return np.argmin(distance, axis=0)


def equals(data1, data2):
    if len(data1) != len(data2):
        return False

    for d1, d2 in izip(data1, data2):
        if d1[0] != d2[0] or d1[1] != d2[1] or d1[2] != d2[2]:
            return False

    return True


def compute_centroids(n_samples, X, n_clusters, centroids):
    new_centroids = np.zeros((n_clusters, 3))
    for i in range(n_clusters):
        Y = X.copy()
        Y = Y[n_samples == i]
        if len(Y) != 0:
            new_centroids[i] = np.transpose(np.average(Y, axis=0)).astype(int)
        else:
            new_centroids[i] = centroids[i]
    return new_centroids


def choose_first_cluster3(X, n_clusters, dist=euclidean_dist):
    y = np.vstack({tuple(row) for row in X})
    random.shuffle(y)
    centroids = np.zeros((n_clusters, 3))
    centroids[0] = y[0]
    for k in range(1, n_clusters):
        d = np.min(find_dist(y, centroids[:k], dist), axis=0)
        d = d/np.sum(d)
        a = np.random.choice(range(y.shape[0]), 1, p=d)
        centroids[k] = y[int(a)]
    return y[:n_clusters]


def choose_first_cluster2(X, n_clusters):
    y = np.vstack({tuple(row) for row in X})
    random.shuffle(y)
    return y[:n_clusters]


def choose_first_cluster1(X, n_clusters):
    y = X.copy()
    random.shuffle(y)
    return y[:n_clusters]


def k_means(X, n_clusters, distance_metric):
    centroids = choose_first_cluster3(X, n_clusters)
    flag = True
    n_samples = np.zeros(X.shape[0])
    n = 0
    #The restriction on n is only triggered when using hist = centroid_histogram(n_samples)
    while flag and n < 100:
        dist = find_dist(X, centroids, distance_metric)
        n_samples = np.transpose(closest_centroid(dist))
        new_centroids = compute_centroids(n_samples, X, n_clusters, centroids)
        flag = not equals(centroids, new_centroids)
        centroids = new_centroids
        n+=1

    return (n_samples, centroids)


def centroid_histogram(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return counts/700


def plot_colors(hist, centroids):
    bar = np.zeros((50, np.sum(hist), 3))
    start_x = 0
    for (percent, color) in zip(hist, centroids):
        end_x = start_x + percent
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50), color.astype("uint8").tolist(), -1)
        start_x = end_x
    return bar


def recolor(image, n_colors):
    X = read_image(image)
    (n_samples, centroids) = k_means(X, n_colors, euclidean_dist)

    img = cv2.imread(image)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = centroids[int(n_samples[i * img.shape[1] + j])]
    cv2.imwrite('superman-batman-min3.png', img)

    hist = centroid_histogram(n_samples)
    cv2.imwrite('hish3.png', plot_colors(hist, centroids))


recolor('superman-batman.png', 16)
