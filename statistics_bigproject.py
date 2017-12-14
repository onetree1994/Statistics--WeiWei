# !/home/onetree/anaconda3/bin python
# -*- coding: utf-8 -*-
# @Time     :   17-12-14 上午8:07
# @Author   :   OneTree
# @FileName :   statistics_bigproject.py
# @Software :   PyCharm
# @Blog     :   http://blog.csdn.net/u013695457?ref=toobar

import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D


# PCA
class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.R = []
        self.w = []
        self.rate = 0

    # find n_components main axises of data X, and return the PCA result
    def fit(self, x, y = []):
        # correlation matrix
        self.R = np.corrcoef(np.transpose(x))
        # eigens and its vetors
        e, ev = np.linalg.eig(self.R)
        rate_base = np.sum(e)
        # sort small->big
        order = np.argsort(e)
        # reverse sort
        order = np.flip(order, 0)
        # set main axises' direction
        self.w = []
        rate = 0
        for i in range(0, self.n_components):
            rate += e[order[i]]
            self.w.append(ev[order[i]])
        self.rate = rate/rate_base
        # calculate PCA result
        pca = np.zeros([len(x), self.n_components])
        for i in range(0, len(x)):
            x_pca = [np.dot(self.w[j], x[i]) for j in range(0, self.n_components)]
            pca[i] = x_pca
        return pca

    # predict a set of new data
    def transform(self, x):
        pca = np.zeros([len(x), self.n_components])
        for i in range(0, len(x)):
            x_pca = [np.dot(self.w[j], x[i]) for j in range(0, self.n_components)]
            pca[i] = x_pca
        return pca


# GMM
class GMM(object):
    def __init__(self):
        self.mu = []
        self.sigma = []
        self.labels = []

    # find n_components gaussian models to mix with MLE
    def fit(self, x, y):
        y_set = set(y)
        self.labels = list(y_set)
        self.mu = []
        self.sigma = []
        for i in y_set:
            mu = 0
            sigma = 0
            xn = x[y == i]
            n = len(xn)
            for j in xn:
                mu += j/n
            for j in xn:
                diff = j - mu
                diff = np.array(diff)
                diff.shape = (len(diff), 1)
                sigma += np.dot(diff, np.transpose(diff))/n
            self.mu.append(mu)
            self.sigma.append(sigma)

    # show the GMM
    def show(self):
        num = 100
        plt.figure(100)
        color = ['ro','bo','yo','go']
        for i in range(0, len(self.mu)):
            x = (np.random.multivariate_normal(self.mu[i], self.sigma[i], num))
            if len(x[0]) == 1:
                plt.plot(x[:,0], [0 for i in range(0, len(x))], color[i])
            else:
                plt.plot(x[:,0], x[:,1], color[i])
        plt.show()

    # predict a new dataset
    def predict(self, x):
        labels = []
        for i in range(0, len(x)):
            label = self.labels[0]
            sigma = np.mat(self.sigma[0])
            sigma_inv = sigma.I
            diff = x[i] - self.mu[0]
            diff = np.array(diff)
            diff.shape = (len(diff), 1)
            label_d = np.transpose(diff)*sigma_inv*diff
            for j in range(1, len(self.labels)):
                sigma = np.mat(self.sigma[j])
                sigma_inv = sigma.I
                diff = x[i] - self.mu[j]
                diff = np.array(diff)
                diff.shape = (len(diff), 1)
                d = np.transpose(diff)*sigma_inv*diff
                if d[0] < label_d[0]:
                    label_d = d
                    label = self.labels[j]
            labels.append(label)
        return labels


# Knife method
def performance(algorithm, X, y):
    n = len(X)
    rate = 0
    for i in range(0, n):
        X_train = []
        X_predict = []
        y_train = []
        y_predict = []
        for j in range(0, n):
            if(i == j):
                X_predict.append(X[j])
                y_predict.append(y[j])
            else:
                X_train.append(X[j])
                y_train.append(y[j])
        algorithm.fit(X_train, y_train)
        x_alg_train = algorithm.transform(X_train)
        x_alg_predict = algorithm.transform(X_predict)
        gmm = GMM()
        gmm.fit(x_alg_train, y_train)
        y_res_predict = gmm.predict(x_alg_predict)
        for j in range(0, len(y_res_predict)):
            if y_res_predict[j] == y_predict[j]:
                rate += 1
    return rate/n


# load iris datasets
iris = datasets.load_iris()
X = iris['data']
X_label = iris['target']
label_set = set(X_label)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(label_set))]

# performance analysis
    # PCA
    # reduct to 2 dimensions
'''pca = PCA(2)
perf = performance(pca, X, X_label)
print("rate:", pca.rate)
print("accuracy:", perf)'''
    # LDA
'''lda = LinearDiscriminantAnalysis(n_components=2)
perf = performance(lda, X, X_label)
print("accuracy:", perf)'''

# show figures
# PCA
    # reduct to 1or2 dimensions
'''pca = PCA(2)
X_res = pca.fit(X)
plt.figure(1)
if len(X_res[0]) == 1:
    for i, color in zip(label_set, colors):
        plt.plot(X_res[X_label == i, 0], [0 for i in range(0, len(X_res[X_label == i]))], 'o', markerfacecolor=tuple(color))
else:
    for i, color in zip(label_set, colors):
        plt.plot(X_res[X_label == i, 0], X_res[X_label == i, 1], 'o', markerfacecolor=tuple(color))
plt.show()'''
    # reduct to 3 dimensions
'''pca = PCA(3)
X_res = pca.fit(X)
plt.figure(2)
ax = plt.subplot(111, projection='3d')
cValue = ['r', 'b', 'g', 'y']
for i, color in zip(label_set, colors):
    plt.scatter(X_res[X_label == i, 0], X_res[X_label == i, 1], X_res[X_label == i, 2], c = cValue[i])
plt.show()'''
# LDA
    # reduct to 2 dimensions
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, X_label)
X_res = lda.transform(X)
plt.figure(3)
if len(X_res[0]) == 1:
    for i, color in zip(label_set, colors):
        plt.plot(X_res[X_label == i, 0], [0 for i in range(0, len(X_res[X_label == i]))], 'o', markerfacecolor=tuple(color))
else:
    for i, color in zip(label_set, colors):
        plt.plot(X_res[X_label == i, 0], X_res[X_label == i, 1], 'o', markerfacecolor=tuple(color))
plt.show()
gmm = GMM()
gmm.fit(X_res, X_label)
gmm.show()