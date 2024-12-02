# src/dimension_reduction/pca.py
from sklearn.decomposition import PCA
import numpy as np

class CustomPCA:
    def __init__(self, n_components=2):
        """
        Initialize PCA dimension reducer
        
        Args:
            n_components (int): Dimension after dimensionality reduction, default is 2
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.explained_variance_ratio_ = None
        
    def fit_transform(self, X):
        """
        Perform PCA dimensionality reduction on data
        Args:
            X: input data matrix
        Returns:
            Data matrix after dimensionality reduction
        """
        reduced_vectors = self.pca.fit_transform(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return reduced_vectors
    
    def transform(self, X):
        """
        Transform new data using trained PCA model
        
        Args:
            X: input data matrix
            
        Returns:
            Data matrix after dimensionality reduction
        """
        return self.pca.transform(X)
    
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio
        
        Returns:
            numpy array: The explained variance ratio of each principal component
        """
        return self.explained_variance_ratio_