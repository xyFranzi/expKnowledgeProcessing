# src/dimension_reduction/pca.py
from sklearn.decomposition import PCA
import numpy as np

class CustomPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.explained_variance_ratio_ = None
        
    def fit_transform(self, X):
        reduced_vectors = self.pca.fit_transform(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return reduced_vectors
    
    def transform(self, X):
        return self.pca.transform(X)
    
    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio_