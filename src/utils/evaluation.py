from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def evaluate_clustering(true_labels, predicted_labels):
    """
    Evaluate clustering results.
    
    Args:
        true_labels: True cluster labels
        predicted_labels: Predicted cluster labels
        
    Returns:
        dict containing evaluation metrics
    """
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    
    return {
        'nmi': nmi_score,
        'ari': ari_score
    }

def analyze_clusters(kmeans, feature_names, n_terms=10):
    """
    Analyze top terms in each cluster.
    
    Args:
        kmeans: Fitted KMeans model
        feature_names: List of feature names
        n_terms: Number of top terms to show
        
    Returns:
        dict mapping cluster indices to top terms
    """
    cluster_terms = {}
    centroids = kmeans.cluster_centers_
    
    for i in range(len(centroids)):
        top_indices = centroids[i].argsort()[::-1][:n_terms]
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_terms[i] = top_terms
        
    return cluster_terms

def find_representative_documents(X, clusters, documents, n_docs=5):
    """
    Find the most representative documents in each cluster
    
    Args:
    X: vector representation (embeddings) of the document
    clusters: the cluster label to which each document belongs
    documents: original document text list
    n_docs: The number of representative documents to return for each cluster
    
    Returns:
        Dictionary: {cluster_id: [doc1, doc2, ...]} 
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    unique_clusters = np.unique(clusters)
    representatives = {}
    
    for cluster_id in unique_clusters:
        # 获取该聚类中的所有文档
        cluster_mask = clusters == cluster_id
        cluster_embeddings = X[cluster_mask]
        cluster_docs = np.array(documents)[cluster_mask]
        
        # 计算聚类中心点(centroid)
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # 计算每个文档与中心点的相似度
        similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1))
        
        # 获取最相似的n_docs个文档
        top_indices = similarities.flatten().argsort()[-n_docs:][::-1]
        representatives[cluster_id] = cluster_docs[top_indices].tolist()
    
    return representatives

def get_cluster_terms_embeddings(texts, clusters, top_n=10, max_features=1000):
    """
    Extract top terms for each cluster using raw text frequency
    
    Args:
        texts: List of original text documents
        clusters: Cluster labels
        top_n: Number of top terms to extract
        max_features: Maximum number of terms to consider
    
    Returns:
        Dictionary mapping cluster IDs to lists of top terms
    """
    # Create a basic count vectorizer
    count_vec = CountVectorizer(max_features=max_features, stop_words='english')
    X_count = count_vec.fit_transform(texts)
    feature_names = count_vec.get_feature_names_out()
    
    cluster_terms = {}
    
    # For each cluster
    for cluster_id in np.unique(clusters):
        # Get documents in this cluster
        cluster_docs = X_count[clusters == cluster_id]
        
        # Calculate average word frequencies for this cluster
        avg_counts = cluster_docs.mean(axis=0).A1
        
        # Get indices of top terms
        top_indices = avg_counts.argsort()[-top_n:][::-1]
        
        # Get the terms
        top_terms = [feature_names[i] for i in top_indices]
        cluster_terms[cluster_id] = top_terms
    
    return cluster_terms