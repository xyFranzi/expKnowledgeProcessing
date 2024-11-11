from sklearn.cluster import KMeans

class DocumentKMeans:
    def __init__(self, n_clusters, random_state=42):
        """
        Initialize K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
        """
        self.kmeans = KMeans(
            n_clusters=n_clusters,  # The number of clusters is equal to the number of categories
            random_state=random_state,
            n_init=10  # Run 10 times to get the best result
        )
        
    def fit_predict(self, X):
        """
        Fit K-means and predict clusters.
        
        Args:
            X: Document-term matrix
            
        Returns:
            array of cluster labels
        """
        return self.kmeans.fit_predict(X)
    
    def get_cluster_centers(self):
        """
        Get cluster centroids.
        
        Returns:
            array of cluster centers
        """
        return self.kmeans.cluster_centers_