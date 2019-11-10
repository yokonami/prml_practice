import numpy as np
class LinearRegression():

    def __init__(self):
        pass

    def fit(self, X, y):
        
        # design matrix large phi
        a = np.hstack((np.ones(shape=(X.shape[0],1)), X))
        
        ata = np.dot(a.T, a)
        aty = np.dot(a.T, y)
        w = np.dot(np.linalg.inv(ata), aty)

        self.coef_ = w[1:]
        self.intercept_ = w[0]
    
    def predict(self, X):
        return np.dot(self.coef_, X.T) + self.intercept_