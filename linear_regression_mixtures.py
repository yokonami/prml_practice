import random
import numpy as np
from scipy.stats import multivariate_normal

class LinearRegressionMixturesEM():
    def __init__(self, K, max_iter=100, tol=1e-4):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        N = len(X)
        K = self.K

        # initialize parameters
        pi = np.array([1 / K] * K)
        w = np.random.randn(K, 1 + X.shape[1])
        beta = 1 / np.std(y)
        rnk = np.zeros(shape=(N, K))

        for iter in range(self.max_iter):

            # E step calc rnk
            phi = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
            for i in range(N):
                norm_const = 0
                xi = phi[i]
                for j in range(K):
                    wjXi = xi.dot(w[j])
                    norm_const += pi[j] * multivariate_normal(wjXi, np.diag([1 / beta])).pdf(y[i])
                for k in range(K):
                    wkXi = xi.dot(w[k])
                    rnk[i, k] = pi[k] * multivariate_normal(wkXi, np.diag([1 / beta])).pdf(y[i]) / norm_const

            # M step
            nk = np.sum(rnk, axis=0)
            pi = nk / N

            # calc w
            for k in range(K):
                atrka = np.dot(phi.T, np.diag(rnk[:, k])).dot(phi)
                atrky = np.dot(phi.T, np.diag(rnk[:, k])).dot(y)
                w[k] = np.linalg.inv(atrka).dot(atrky).flatten()

            # calc beta
            beta = 0
            for k in range(K):
                beta += phi.dot(w[k]).dot(np.diag(rnk[:, k])).dot(phi.dot(w[k]))
            beta = N * (1 / beta)

            # calc likelihood
            likelihood = 0
            for i in range(N):
                likelihood_i = 0
                xi = phi[i]
                for j in range(K):
                    wjXi = xi.dot(w[j])
                    likelihood_i += pi[j] * multivariate_normal(wjXi, np.diag([1 / beta])).pdf(y[i])
                likelihood += np.log(likelihood_i)

            if iter != 0:
                if abs(prev_likelihood - likelihood) < self.tol:
                    break
            prev_likelihood = likelihood

            if iter == self.max_iter:
                break

        self.w = w
        self.beta = beta
        self.pi = pi
        self.rnk = rnk

    def predict(self, X):
        phi = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        y = np.zeros(shape=len(X))
        for k in range(self.K):
            y += self.pi[k] * phi.dot(self.w[k].T)

        return y

    def predictive_distribution(self, x):

        w, beta = self.w, self.beta
        phi = np.hstack((np.ones(shape=(1, 1)), x.reshape(1, len(x))))
        f_list = []
        for k in range(self.K):
            f_list.append(multivariate_normal(phi.dot(w[k]), np.diag([1 / beta])).pdf)

        return lambda x: sum([p*f(x) for p, f in zip(self.pi, f_list)])