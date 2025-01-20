from sklearn.cluster import KMeans

class DocumentKMeans:
    def __init__(self, n_clusters, random_state=42):

        self.kmeans = KMeans(
            n_clusters=n_clusters,  # The number of clusters is equal to the number of categories
            random_state=random_state,
            n_init=10  # Run 10 times to get the best result
        )
        
    def fit_predict(self, X):

        return self.kmeans.fit_predict(X)
    
    def get_cluster_centers(self):

        return self.kmeans.cluster_centers_