from src.text_vectorization.tfidf import CustomTfidfVectorizer
from src.text_vectorization.fasttext_vec import CustomFastTextVectorizer
from src.clustering.kmeans import DocumentKMeans
from src.utils.evaluation import evaluate_clustering, analyze_clusters
from src.utils.preprocess import load_dataset
from src.utils.viz import visualize_clustering_results
import pickle
import os

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