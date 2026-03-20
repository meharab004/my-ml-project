import numpy as np

def predict(X):
    weights = np.array([0.5, -0.3, 0.8])
    return X @ weights

def get_accuracy():
    return 0.50
