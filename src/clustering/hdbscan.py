import hdbscan

class DocumentHDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None, metric='euclidean'):
        """
        Initialize HDBSCAN clustering.
        
        Args:
            min_cluster_size: The smallest size grouping that can be considered a cluster.
            min_samples: The number of samples in a neighborhood for a point to be a core point.
            metric: The metric to use when calculating distance between instances.
        """
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            core_dist_n_jobs=-1  # Use all CPU cores
        )
        self.labels_ = None

    def fit_predict(self, X):
        """
        Fit HDBSCAN and predict clusters.
        
        Args:
            X: Document-term matrix
            
        Returns:
            array of cluster labels (-1 indicates noise points)
        """
        self.labels_ = self.hdbscan.fit_predict(X)
        return self.labels_

    def get_cluster_info(self):
        """
        Get clustering information.
        
        Returns:
            dict containing:
                - number of clusters (excluding noise)
                - number of noise points
        """
        if self.labels_ is None:
            raise ValueError("Must call fit_predict before getting cluster info")
            
        return {
            'n_clusters': len(set(self.labels_)) - (1 if -1 in self.labels_ else 0),
            'n_noise': list(self.labels_).count(-1),
        }
