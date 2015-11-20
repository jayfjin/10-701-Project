from sklearn import linear_model
import numpy as np

def lasso_select(X, Y, alpha=0.01):
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(X, Y)
    return np.where(lasso.coef_ != 0)[0]
