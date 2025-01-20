import hdbscan

class DocumentHDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None, metric='euclidean'):
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            core_dist_n_jobs=-1  # Use all CPU cores
        )
        self.labels_ = None

    def fit_predict(self, X):
        self.labels_ = self.hdbscan.fit_predict(X)
        return self.labels_

    def get_cluster_info(self):
        if self.labels_ is None:
            raise ValueError("Must call fit_predict before getting cluster info")
            
        return {
            'n_clusters': len(set(self.labels_)) - (1 if -1 in self.labels_ else 0),
            'n_noise': list(self.labels_).count(-1),
        }
