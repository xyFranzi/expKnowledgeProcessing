from sklearn.cluster import DBSCAN

class DocumentDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=-1  # Use all CPU cores
        )
        self.labels_ = None
        
    def fit_predict(self, X):
        self.labels_ = self.dbscan.fit_predict(X)
        return self.labels_
    
    def get_cluster_info(self):
        if self.labels_ is None:
            raise ValueError("Must call fit_predict before getting cluster info")
            
        return {
            'n_clusters': len(set(self.labels_)) - (1 if -1 in self.labels_ else 0),
            'n_noise': list(self.labels_).count(-1),
            'core_samples_mask': self.dbscan.core_sample_indices_
        }
