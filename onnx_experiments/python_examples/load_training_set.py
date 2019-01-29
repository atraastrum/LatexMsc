from sklearn import datasets
import numpy as np

def get():
    diabetes = datasets.load_diabetes()
    features = diabetes.data
    labels = diabetes.target

    return {'features': features, 'labels': labels.reshape(-1, 1)}

