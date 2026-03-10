import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    x_sample,x_feat=X.shape
    w=np.zeros(x_feat)
    b=0
    for _ in range(steps):
        linear=np.dot(X,w)+b
        p=_sigmoid(linear)
        error=p-y
        dw=(1/x_sample)*(np.dot(X.T,error))
        db=(1/x_sample)*np.sum(error)
        w-=lr*dw
        b-=lr*db
    # Write code here
    return w,b
    pass