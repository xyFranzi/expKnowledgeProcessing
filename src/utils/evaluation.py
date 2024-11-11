from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

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