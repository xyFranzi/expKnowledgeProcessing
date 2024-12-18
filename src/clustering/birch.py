from sklearn.cluster import Birch

class DocumentBirch:
    def __init__(self, n_clusters=None):
        """
        Initialize Birch clustering.

        Args:
            n_clusters: Number of clusters (None for automatic threshold-based clustering)
        """
        self.birch = Birch(n_clusters=n_clusters)

    def fit_predict(self, X):
        """
        Fit Birch and predict clusters.

        Args:
            X: Document-term matrix

        Returns:
            array of cluster labels
        """
        return self.birch.fit_predict(X)

    def get_subcluster_centers(self):
        """
        Get centroids of subclusters.

        Returns:
            array of subcluster centers
        """
        return self.birch.subcluster_centers_

    def get_n_subclusters(self):
        """
        Get the number of subclusters formed.

        Returns:
            int: Number of subclusters
        """
        return len(self.birch.subcluster_centers_)
