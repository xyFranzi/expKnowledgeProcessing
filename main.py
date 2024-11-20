from src.utils.preprocess import load_dataset
import os
import time
from datetime import datetime
import numpy as np
from src.utils.preprocess import load_dataset
from src.utils.experiment import ExperimentManager

def main():
    # 加载数据
    documents = load_dataset()
    print("Dataset loaded successfully.")
    
    # Record start time
    start_time = time.time()
    
    # 设置并运行实验
    manager = ExperimentManager()
    manager.setup_experiments(n_clusters=len(documents.target_names))
    manager.run_all(documents)
    
    # 可视化比较
    manager.visualize_results()
    
    # 输出评估指标比较
    metrics_df = manager.compare_metrics()
    print("\nEvaluation Metrics:")
    print(metrics_df)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/clustering_results_{timestamp}.txt'
    os.makedirs('results', exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("=== Document Clustering Experiments Results ===\n\n")
        f.write("Evaluation Metrics:\n")
        f.write(metrics_df.to_string())
        f.write("\n\n")
        
        # 保存详细结果
        for experiment_name, result in manager.results.items():
            f.write(f"\n=== {experiment_name.upper()} ===\n")
            
            # 保存聚类分布
            clusters = result['clusters']
            unique, counts = np.unique(clusters, return_counts=True)
            f.write("\nCluster Distribution:\n")
            for cluster, count in zip(unique, counts):
                f.write(f"Cluster {cluster}: {count} documents\n")
            
            # 保存每个簇的主要词语
            f.write("\nTop Terms per Cluster:\n")
            for cluster_id, terms in result['cluster_terms'].items():
                f.write(f"\nCluster {cluster_id}:\n")
                f.write(", ".join(terms[:10]))
                f.write("\n")
        
        # 保存执行时间
        total_time = time.time() - start_time
        f.write(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print(f"\nResults saved to {results_file}")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
'''
###  最初的
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