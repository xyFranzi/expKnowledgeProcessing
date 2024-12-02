from sklearn.decomposition import TruncatedSVD
import numpy as np

class CustomSVD:
    def __init__(self, n_components=2):
        """
        Initialize SVD dimension reducer
        
        Args:
            n_components (int): Dimension after dimensionality reduction, default is 2
        """
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.explained_variance_ratio_ = None
        
    def fit_transform(self, X):
        """
        Perform SVD dimensionality reduction on data
        
        Args:
            X: input data matrix (can be sparse)
            
        Returns:
            Data matrix after dimensionality reduction
        """
        reduced_vectors = self.svd.fit_transform(X)
        self.explained_variance_ratio_ = self.svd.explained_variance_ratio_
        return reduced_vectors
    
    def transform(self, X):
        """
        Transform new data using trained SVD model
        
        Args:
            X: input data matrix
            
        Returns:
            Data matrix after dimensionality reduction
        """
        return self.svd.transform(X)
    
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio
        
        Returns:
            numpy array: The explained variance ratio of each component
        """
        return self.explained_variance_ratio_