from src.text_vectorization.tfidf import CustomTfidfVectorizer
from src.text_vectorization.fasttext_vec import CustomFastTextVectorizer
from src.text_vectorization.minilm_vec import CustomMiniLMVectorizer
from src.clustering.kmeans import DocumentKMeans
from src.utils.evaluation import evaluate_clustering, analyze_clusters,find_representative_documents,get_cluster_terms_embeddings
from src.utils.preprocess import load_dataset
from src.utils.viz import visualize_clustering_results
import pickle
import os
import time
from datetime import datetime

def experiment_tfidf(dataset):
    """Experiment 1: TF-IDF + K-means"""
    print("\n=== Experiment 1: TF-IDF + K-means ===")
    
    # Vectorization
    vectorizer = CustomTfidfVectorizer()
    X = vectorizer.fit_transform(dataset.data)
    print("TF-IDF vectorization complete.")
    
    # Clustering
    kmeans = DocumentKMeans(n_clusters=len(dataset.target_names))
    clusters = kmeans.fit_predict(X)
    
    # Evaluation
    results = evaluate_clustering(dataset.target, clusters)
    print("\nTF-IDF + K-means Results:")
    print(f"NMI Score: {results['nmi']:.3f}")
    print(f"ARI Score: {results['ari']:.3f}")
    
    # Analyze top terms per cluster
    feature_names = vectorizer.get_feature_names()
    cluster_terms = analyze_clusters(kmeans.kmeans, feature_names)
    
    print("\nTop terms per cluster:")
    for cluster_id, terms in cluster_terms.items():
        print(f"\nCluster {cluster_id}:")
        print(", ".join(terms))
    
    # Visualization
    visualize_clustering_results(X, clusters, cluster_terms,title="TF-IDF + K-means Clustering")
    
    return results

def experiment_fasttext(dataset):
    """Experiment 2: FastText + K-means"""

    print("\n=== Experiment 2: FastText + K-means ===")
    
    # Load or create vectors

    vector_file = 'fasttext_vectors.pkl'
    if os.path.exists(vector_file):
        with open(vector_file, 'rb') as f:
            X = pickle.load(f)
        print("Loaded FastText vectors from file.")

    else:
        print("----create new----")
        vectorizer = CustomFastTextVectorizer(
            #model_path='models/cc.en.300.bin'
            model_path = '/Users/yue/Documents/code/expKnowledgeProcessing/models/cc.en.300.bin'
        )
        X = vectorizer.fit_transform(dataset.data)
        with open(vector_file, 'wb') as f:
            pickle.dump(X, f)
        print("FastText vectorization complete and saved.")
    
    # Clustering
    kmeans = DocumentKMeans(n_clusters=len(dataset.target_names))
    clusters = kmeans.fit_predict(X)
    
    # Evaluation
    results = evaluate_clustering(dataset.target, clusters)
    print("\nFastText + K-means Results:")
    print(f"NMI Score: {results['nmi']:.3f}")
    print(f"ARI Score: {results['ari']:.3f}")
    
    
    # Find representative documents
    representatives = find_representative_documents(X, clusters, dataset.data)
    print("\nRepresentative documents per cluster:")
    for cluster_id, docs in representatives.items():
        print(f"\nCluster {cluster_id}:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"{i}. {doc[:200]}...")
    
    
    # Visualization
    visualize_clustering_results(X, clusters, title="FastText + K-means Clustering")
    
    return results

def experiment_minilm(dataset):
    """Experiment 3: MiniLM + K-means"""
    print("\n=== Experiment 3: MiniLM + K-means ===")
    
    # Vectorization
    vectorizer = CustomMiniLMVectorizer()
    X = vectorizer.fit_transform(dataset.data)
    print("MiniLM vectorization complete.")
    
    # Clustering
    kmeans = DocumentKMeans(n_clusters=len(dataset.target_names))
    clusters = kmeans.fit_predict(X)
    
    # Evaluation
    results = evaluate_clustering(dataset.target, clusters)
    print("\nMiniLM + K-means Results:")
    print(f"NMI Score: {results['nmi']:.3f}")
    print(f"ARI Score: {results['ari']:.3f}")
    
    # Find representative documents
    representatives = find_representative_documents(X, clusters, dataset.data)
    print("\nRepresentative documents per cluster:")
    for cluster_id, docs in representatives.items():
        print(f"\nCluster {cluster_id}:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"{i}. {doc[:200]}...")
    
    # Get cluster terms using frequency-based approach
    cluster_terms = get_cluster_terms_embeddings(
        texts=dataset.data,
        clusters=clusters,
        top_n=10
    )
    
    # Print top terms for each cluster
    print("\nTop terms per cluster:")
    for cluster_id, terms in cluster_terms.items():
        print(f"\nCluster {cluster_id}:")
        print(", ".join(terms))
    
    # Visualization with cluster terms
    visualize_clustering_results(X, clusters, cluster_terms, title="MiniLM + K-means Clustering")

    return results
def main():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Load dataset
    dataset = load_dataset()
    print("Dataset loaded successfully.")
    
    # Record start time
    start_time = time.time()
    
    # Run all experiments
    results = {
        #'tfidf': experiment_tfidf(dataset),
        'fasttext': experiment_fasttext(dataset),
        #'minilm': experiment_minilm(dataset)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/clustering_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("=== Document Clustering Experiments Results ===\n\n")
        for method, scores in results.items():
            f.write(f"\n{method.upper()} + K-means:\n")
            f.write(f"NMI Score: {scores['nmi']:.3f}\n")
            f.write(f"ARI Score: {scores['ari']:.3f}\n")
        
        total_time = time.time() - start_time
        f.write(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print(f"\nResults saved to {results_file}")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()



'''
def main():
    # Load dataset
    dataset = load_dataset()
    print("Dataset loaded.")
    
    # Initialize and fit TF-IDF
    vectorizer_A = CustomTfidfVectorizer() # 重命名为方法A了
    # X = vectorizer_A.fit_transform(dataset.data) # 暂时注释掉

    
    # Vectorising with Fast Text
    # Load previous vectorization file (if there is)
    if os.path.exists('vectorized_data.pkl'):
        with open('vectorized_data.pkl', 'rb') as f:
            X = pickle.load(f)
        print("Loaded vectorized data from file.")
    else:
    # Perform vectorization as before
        vectorizer_B = CustomFastTextVectorizer(model_path='D:/mypython/KP/expKnowledgeProcessing/models/cc.en.300.bin') 
                    # have to use absolute path otherwise bug appears i dont know why :(
        print("Vectorizing dataset...")
        X = vectorizer_B.fit_transform(dataset.data)
        with open('vectorized_data.pkl', 'wb') as f:
            pickle.dump(X, f)
        print("Vectorization complete and data saved.")
    

    # Vectorising with Minilm
    vectorizer_C = MiniLMVectorizer()
    X = vectorizer.fit_transform(dataset.data)
    
    # Perform clustering
    print("Performing KMeans clustering...") # to track the running process
    kmeans = DocumentKMeans(n_clusters=len(dataset.target_names))
    clusters = kmeans.fit_predict(X)
    print("Clustering complete.")
    
    # Evaluate results
    print("Evaluating clustering results...")
    evaluation_results = evaluate_clustering(dataset.target, clusters)
    print("Clustering Evaluation:")
    print(f"NMI Score: {evaluation_results['nmi']:.3f}")
    print(f"ARI Score: {evaluation_results['ari']:.3f}")
    
    # Analyze clusters # 这部分还是第一个方法的 还没改完
    feature_names = vectorizer_A.get_feature_names()
    cluster_terms = analyze_clusters(kmeans.kmeans, feature_names)
    
    print("\nTop terms per cluster:")
    for cluster_id, terms in cluster_terms.items():
        print(f"\nCluster {cluster_id}:")
        print(", ".join(terms))

    # Visualize results
    visualize_clustering_results(X, clusters, cluster_terms)

if __name__ == "__main__":
    main()

'''