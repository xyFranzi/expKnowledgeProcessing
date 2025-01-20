from sklearn.decomposition import TruncatedSVD
import numpy as np

class CustomSVD:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.explained_variance_ratio_ = None
        
    def fit_transform(self, X):
        reduced_vectors = self.svd.fit_transform(X)
        self.explained_variance_ratio_ = self.svd.explained_variance_ratio_
        return reduced_vectors
    
    def transform(self, X):
        return self.svd.transform(X)
    
    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio_