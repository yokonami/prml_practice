import random
import numpy as np
class KMeans():

    def __init__(self, n_class, max_iter=100, tol=1e-4):
        self.n_class = n_class
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n = len(X)
        n_class = self.n_class
        
        # initialize mu_k by randomly selected data points
        mu = X[random.sample(list(range(n)), n_class)]
        
        for iter in range(self.max_iter):
            # E step calc rnk
            
            # distance matrix between each data points and mu_k
            dist_nk = np.zeros(shape=(n, n_class))
            for i in range(n):
                for k in range(n_class):
                    dist_nk[i, k] = np.linalg.norm(X[i] - mu[k])
            
            # cluster assignment vector
            rnk = np.argmin(dist_nk, axis=1)

            # M step calc mu
            for k in range(n_class):
                mu[k] = np.sum(X[rnk==k], axis=0) / sum(rnk==k)
                
            # calc loss function
            j = 0
            for k in range(n_class):
                xk = X[rnk==k]
                for i in range(len(xk)):
                    j += np.linalg.norm(xk-mu[k])**2

            if iter != 0:
                if abs(prev_j - j) < self.tol:
                    break
            prev_j = j

        self.mu = mu
        self.rnk = rnk
        self.j = j
        
    def predict(self, X):
        n = len(X)
        n_class = self.n_class

        # calc rnk
        # distance matrix between each data points and mu_k
        dist_nk = np.zeros(shape=(n, n_class))
        for i in range(n):
            for k in range(n_class):
                dist_nk[i, k] = np.linalg.norm(X[i] - self.mu[k])

        # cluster assignment vector
        rnk = np.argmin(dist_nk, axis=1)

        return rnk