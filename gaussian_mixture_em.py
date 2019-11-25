import random
import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureEM():
    def __init__(self, n_class, max_iter=100, tol=1e-4):
        self.n_class = n_class
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n = len(X)
        n_class = self.n_class

        # initialize parameters
        mu = X[random.sample(list(range(n)), n_class)]
        pi = np.array([1 / n_class] * n_class)
        cov = np.array([np.cov(X, rowvar=False)] * n_class)

        for iter in range(self.max_iter):
            # E step calc rnk

            rnk = np.zeros(shape=(n, n_class))
            for i in range(n):
                norm_const = 0
                for k in range(n_class):
                    norm_const += pi[k] * multivariate_normal(mu[k], cov[k]).pdf(X[i])
                for k in range(n_class):
                    rnk[i, k] = pi[k] * multivariate_normal(mu[k], cov[k]).pdf(X[i]) / norm_const


            # M step
            nk = np.sum(rnk, axis=0)
            pi = nk / n

            for k in range(n_class):
                mu[k] = np.sum(rnk[:, k].reshape(n, 1) * X, axis=0) / nk[k]

            for k in range(n_class):
                cov[k] = (X - mu[k]).T.dot(np.diag(rnk[:, k])).dot(X - mu[k]) / nk[k]

            # calc loss function
            j = 0
            mn_list = [multivariate_normal(mu[k], cov[k]) for k in range(n_class)]
            for i in range(n):
                jk = 0
                for k in range(n_class):
                    jk += pi[k] * mn_list[k].pdf(X[i])
                j += np.log(jk)

            if iter != 0:
                if abs(prev_j - j) < self.tol:
                    break
            prev_j = j

            if iter == self.max_iter:
                break

        self.mu = mu
        self.cov = cov
        self.pi = pi
        self.rnk = rnk
        self.j = j

    def predict(self, X):
        n = len(X)
        n_class = self.n_class
        pi = self.pi
        cov = self.cov
        mu = self.mu

        rnk = np.zeros(shape=(n, n_class))
        for i in range(n):
            norm_const = 0
            for k in range(n_class):
                norm_const += pi[k] * multivariate_normal(mu[k], cov[k]).pdf(X[i])
            for k in range(n_class):
                rnk[i, k] = pi[k] * multivariate_normal(mu[k], cov[k]).pdf(X[i]) / norm_const

        return rnk
