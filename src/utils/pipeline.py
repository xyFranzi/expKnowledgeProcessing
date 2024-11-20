from src.text_vectorization.tfidf import CustomTfidfVectorizer
from src.text_vectorization.fasttext_vec import CustomFastTextVectorizer
from src.text_vectorization.minilm_vec import CustomMiniLMVectorizer
from src.clustering.kmeans import DocumentKMeans
from src.clustering.dbscan import DocumentDBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os

class DocumentClusteringPipeline:
    def __init__(self, vectorizer_name, clusterer_name, n_clusters=None):
        self.vectorizer_name = vectorizer_name
        self.clusterer_name = clusterer_name
        self.n_clusters = n_clusters
        self.results = {}  
        self.pca = PCA(n_components=2)  
        self.setup_components()
        
    def setup_components(self):
        # Set vectorization method
        if self.vectorizer_name == 'tfidf':
            self.vectorizer = CustomTfidfVectorizer()
        elif self.vectorizer_name == 'fasttext':
            self.vectorizer = CustomFastTextVectorizer(
                    # model_path='/Users/yue/Documents/code/expKnowledgeProcessing/models/cc.en.300.bin'
                model_path='D:/mypython/KP/expKnowledgeProcessing/models/cc.en.300.bin'
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
    
    def process(self, documents):
        # Store original data set
        self.results['original_dataset'] = documents
        
        # 向量化
        #  Load previous vectorization file (if there is)
                
        if self.vectorizer_name == 'fasttext':
            if os.path.exists('vectorized_data.pkl'):
                with open('vectorized_data.pkl', 'rb') as f:
                    vectors = pickle.load(f)
                print("Loaded vectorized data from file.")
            else:
            # Perform vectorization as before
                print("Vectorizing dataset...")
                vectors = self.vectorizer.fit_transform(documents.data)
                with open('vectorized_data.pkl', 'wb') as f:
                    pickle.dump(vectors, f)
                print("Vectorization complete and data saved.")
        else:
            vectors = self.vectorizer.fit_transform(documents.data)
            
        self.results['vectors'] = vectors

        # Dimensionality reduction for visualization
        if hasattr(vectors, 'toarray'):
            dense_vectors = vectors.toarray()
        else:
            dense_vectors = vectors
        
        reduced_vectors = self.pca.fit_transform(dense_vectors)
        self.results['reduced_vectors'] = reduced_vectors

        # clustering
        clusters = self.clusterer.fit_predict(vectors)
        self.results['clusters'] = clusters
        
        # Evaluate
        metrics = self.evaluate(clusters)
        self.results['metrics'] = metrics
        
        return self.results

    def evaluate(self, clusters):
        """
        Evaluate clustering results, including internal and external evaluation metrics
        
        Returns:
            dict containing various evaluation metrics
        """
        true_labels = self.results['original_dataset'].target
        vectors = self.results['vectors']
        
        if not isinstance(vectors, np.ndarray):
            vectors = vectors.toarray()
            
        metrics = {
            # External evaluation metrics (requires real labels)
            'nmi': normalized_mutual_info_score(true_labels, clusters),
            'ari': adjusted_rand_score(true_labels, clusters),
            
            # Internal evaluation metrics (no real labels required)
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

        Args:
            texts: original text list
            clusters: cluster labels
            top_n: the number of words returned by each cluster
            max_features: Maximum number of words considered
        
        Returns:
            Dictionary：{cluster_id: [term1, term2, ...]}
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

        Args:
            X: vector representation of the document
            clusters: cluster labels
            documents: original document list
            n_docs: Number of documents returned for each cluster
        
        Returns:
            Dictionary：{cluster_id: [doc1, doc2, ...]}
        """
        unique_clusters = np.unique(clusters)
        representatives = {}
        
        for cluster_id in unique_clusters:
            # Get the documents for this cluster
            cluster_mask = clusters == cluster_id
            cluster_embeddings = X[cluster_mask]
            cluster_docs = np.array(documents)[cluster_mask]
            
            # Calculate cluster center
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate similarity
            similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1))
            
            # Get the most similar document
            top_indices = similarities.flatten().argsort()[-n_docs:][::-1]
            representatives[cluster_id] = cluster_docs[top_indices].tolist()
        
        return representatives

    def analyze_clusters(self, kmeans, feature_names, n_terms=10):
        """
        Analyze the main feature words of each cluster (only applicable to K-means clustering)
        
        Args:
            kmeans: trained KMeans model
            feature_names: list of feature names
            n_terms: Number of terms returned
        
        Returns:
            Dictionary：{cluster_id: [term1, term2, ...]}
        """
        cluster_terms = {}
        centroids = kmeans.cluster_centers_
        
        for i in range(len(centroids)):
            top_indices = centroids[i].argsort()[::-1][:n_terms]
            top_terms = [feature_names[idx] for idx in top_indices]
            cluster_terms[i] = top_terms
            
        return cluster_terms