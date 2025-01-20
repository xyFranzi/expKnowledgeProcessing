from sklearn.cluster import Birch

class DocumentBirch:
    def __init__(self, n_clusters=None):
        self.birch = Birch(n_clusters=n_clusters)

    def fit_predict(self, X):
        return self.birch.fit_predict(X)

    def get_subcluster_centers(self):
        return self.birch.subcluster_centers_

    def get_n_subclusters(self):
        return len(self.birch.subcluster_centers_)
