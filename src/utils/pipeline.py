from src.dimension_reduction.pca import CustomPCA
from src.text_vectorization.tfidf import CustomTfidfVectorizer
from src.text_vectorization.fasttext_vec import CustomFastTextVectorizer 
from src.text_vectorization.minilm_vec import CustomMiniLMVectorizer
from src.clustering.kmeans import DocumentKMeans
from src.clustering.dbscan import DocumentDBSCAN
from src.clustering.hdbscan import DocumentHDBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os

class DocumentClusteringPipeline:
    def __init__(self, vectorizer_name, clusterer_name, n_clusters=None, n_components=None, variance_threshold=0.95):
        """
        Initialize the document clustering pipeline
        
        参数:
            vectorizer_name (str): vectorization method ('tfidf', 'fasttext', 或 'minilm')
            clusterer_name (str): clustering method ('kmeans', 'dbscan', 或 'hdbscan')
            n_clusters (int, optional): Number of clusters for K-means clustering. Can be None for DBSCAN and HDBSCAN
            n_components (int, optional): The number of dimensions for PCA dimensionality reduction. If None, automatically select
            variance_threshold (float, optional): The variance threshold when automatically selecting dimensions, the default is 0.95, which means 95% of the variance is retained
        """
        self.vectorizer_name = vectorizer_name
        self.clusterer_name = clusterer_name
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.results = {}
        self.pca_vis = PCA(n_components=2)# for visualization
        
        # If n_components is specified, initialize PCA
        if self.n_components is not None:
            self.pca = CustomPCA(n_components=self.n_components)
        else:
            self.pca = None  # Will be automatically set based on data in the process method
        self.setup_components()   
    
    def setup_components(self):
        # Set vectorization method
        if self.vectorizer_name == 'tfidf':
            self.vectorizer = CustomTfidfVectorizer()
        elif self.vectorizer_name == 'fasttext':
            self.vectorizer = CustomFastTextVectorizer(
                model_path='/Users/yue/Documents/code/expKnowledgeProcessing/models/cc.en.300.bin'
            )
        elif self.vectorizer_name == 'minilm':
            self.vectorizer = CustomMiniLMVectorizer()
        
        # Set clustering method
        if self.clusterer_name == 'kmeans':
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for KMeans clustering")
            self.clusterer = DocumentKMeans(n_clusters=self.n_clusters)
        elif self.clusterer_name == 'dbscan':
            self.clusterer = DocumentDBSCAN()
        elif self.clusterer_name == 'hdbscan':
            self.clusterer = DocumentHDBSCAN()
    
    def select_n_components(self, vectors):
            """
            Automatically select the number of dimensions for PCA dimensionality reduction
            Minimum dimension required to reach threshold based on cumulative explained variance ratio
            """
            # If the vector is a sparse matrix, convert it to a dense matrix
            if hasattr(vectors, 'toarray'):
                vectors = vectors.toarray()
                
            # First fit the data with full PCA
            temp_pca = PCA()
            temp_pca.fit(vectors)
            
            # Calculate cumulative variance ratio
            cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            
            # Find the first dimension that exceeds the threshold
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            
            print(f"Selected {n_components} components (explaining {cumsum[n_components-1]:.2%} of variance)")
            return n_components

    def process(self, documents):
        # Store original dataset
        self.results['original_dataset'] = documents
        
        # Vectorization
        if self.vectorizer_name == 'fasttext':
            if os.path.exists('vectorized_data.pkl'):
                with open('vectorized_data.pkl', 'rb') as f:
                    vectors = pickle.load(f)
                print("Loaded vectorized data from file.")
            else:
                print("Vectorizing dataset...")
                vectors = self.vectorizer.fit_transform(documents.data)
                with open('vectorized_data.pkl', 'wb') as f:
                    pickle.dump(vectors, f)
                print("Vectorization complete and data saved.")
        else:
            vectors = self.vectorizer.fit_transform(documents.data)
            
        self.results['original_vectors'] = vectors

        # Convert sparse matrix to dense if necessary
        if hasattr(vectors, 'toarray'):
            dense_vectors = vectors.toarray()
        else:
            dense_vectors = vectors
    
        # If n_components is not specified, dimensions are automatically selected
        if self.n_components is None:
            self.n_components = self.select_n_components(dense_vectors)
            
        # Initialize PCA with selected dimensions
        self.pca = CustomPCA(n_components=self.n_components)
        
        # Dimensionality reduction for clustering
        clustering_vectors = self.pca.fit_transform(dense_vectors)
        self.results['reduced_vectors'] = clustering_vectors
        
        # Calculate the effect index of dimensionality reduction
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance_ratio = np.sum(explained_variance_ratio)
        self.results['dim_reduction_metrics'] = {
            'n_components': self.n_components,
            'explained_variance_ratio': cumulative_variance_ratio,
            'component_variance_ratios': explained_variance_ratio.tolist()
        }
        
        # Always do 2D reduction for visualization
        vis_vectors = self.pca_vis.fit_transform(dense_vectors)
        self.results['visualization_vectors'] = vis_vectors

        # Clustering on appropriate vectors
        clusters = self.clusterer.fit_predict(clustering_vectors)
        self.results['clusters'] = clusters

        # Evaluate
        metrics = self.evaluate(clusters)
        self.results['metrics'] = metrics
        
        return self.results

    def evaluate(self, clusters):
        true_labels = self.results['original_dataset'].target
        vectors = self.results['reduced_vectors']  # Use reduced vectors for evaluation
            
        metrics = {
            'nmi': normalized_mutual_info_score(true_labels, clusters),
            'ari': adjusted_rand_score(true_labels, clusters),
            'silhouette': silhouette_score(vectors, clusters),
            'calinski': calinski_harabasz_score(vectors, clusters)
        }
        
        # Analyze clustering results
        self.results['cluster_terms'] = self.get_cluster_terms_embeddings(
            self.results['original_dataset'].data,
            clusters
        )
        
        # Find representative documents
        self.results['representative_docs'] = self.find_representative_documents(
            vectors,
            clusters,
            self.results['original_dataset'].data
        )
        
        return metrics

    def get_cluster_terms_embeddings(self, texts, clusters, top_n=10, max_features=1000):
        """
        Extract the main words of each cluster
        """
        count_vec = CountVectorizer(max_features=max_features, stop_words='english')
        X_count = count_vec.fit_transform(texts)
        feature_names = count_vec.get_feature_names_out()
        
        cluster_terms = {}
        
        for cluster_id in np.unique(clusters):
            cluster_docs = X_count[clusters == cluster_id]
            avg_counts = cluster_docs.mean(axis=0).A1
            top_indices = avg_counts.argsort()[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            cluster_terms[cluster_id] = top_terms
        
        return cluster_terms

    def find_representative_documents(self, X, clusters, documents, n_docs=5):
        """
        Find the most representative documents in each cluster
        """
        unique_clusters = np.unique(clusters)
        representatives = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_embeddings = X[cluster_mask]
            cluster_docs = np.array(documents)[cluster_mask]
            
            centroid = np.mean(cluster_embeddings, axis=0)
            similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1))
            
            top_indices = similarities.flatten().argsort()[-n_docs:][::-1]
            representatives[cluster_id] = cluster_docs[top_indices].tolist()
        
        return representatives