from src.text_vectorization.tfidf import CustomTfidfVectorizer
from src.clustering.kmeans import DocumentKMeans
from src.utils.evaluation import evaluate_clustering, analyze_clusters
from src.utils.preprocess import load_dataset
from src.utils.viz import visualize_clustering_results

def main():
    # Load dataset
    dataset = load_dataset()
    
    # Initialize and fit TF-IDF
    vectorizer = CustomTfidfVectorizer()
    X = vectorizer.fit_transform(dataset.data)
    
    # Perform clustering
    kmeans = DocumentKMeans(n_clusters=len(dataset.target_names))
    clusters = kmeans.fit_predict(X)
    
    # Evaluate results
    evaluation_results = evaluate_clustering(dataset.target, clusters)
    print("Clustering Evaluation:")
    print(f"NMI Score: {evaluation_results['nmi']:.3f}")
    print(f"ARI Score: {evaluation_results['ari']:.3f}")
    
    # Analyze clusters
    feature_names = vectorizer.get_feature_names()
    cluster_terms = analyze_clusters(kmeans.kmeans, feature_names)
    
    print("\nTop terms per cluster:")
    for cluster_id, terms in cluster_terms.items():
        print(f"\nCluster {cluster_id}:")
        print(", ".join(terms))

    # Visualize results
    visualize_clustering_results(X, clusters, cluster_terms)

if __name__ == "__main__":
    main()