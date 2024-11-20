from sklearn.cluster import DBSCAN

class DocumentDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Initialize DBSCAN clustering.
        
        Args:
            eps: The maximum distance between two samples for one to be considered
                as in the neighborhood of the other
            min_samples: The minimum number of samples in a neighborhood for a point
                to be considered as a core point
            metric: The metric to use for distance computation
        """
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=-1  # Use all CPU cores
        )
        self.labels_ = None
        
    def fit_predict(self, X):
        """
        Fit DBSCAN and predict clusters.
        
        Args:
            X: Document-term matrix
            
        Returns:
            array of cluster labels (-1 indicates noise points)
        """
        self.labels_ = self.dbscan.fit_predict(X)
        return self.labels_
    
    def get_cluster_info(self):
        """
        Get clustering information.
        
        Returns:
            dict containing:
                - number of clusters (excluding noise)
                - number of noise points
                - core samples mask
        """
        if self.labels_ is None:
            raise ValueError("Must call fit_predict before getting cluster info")
            
        return {
            'n_clusters': len(set(self.labels_)) - (1 if -1 in self.labels_ else 0),
            'n_noise': list(self.labels_).count(-1),
            'core_samples_mask': self.dbscan.core_sample_indices_
        }
