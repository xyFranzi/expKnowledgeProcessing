from src.utils.preprocess import load_dataset
import os
import time
from datetime import datetime
import numpy as np
from src.utils.preprocess import load_dataset
from src.utils.experiment import ExperimentManager

def main():
    # load dataset
    documents = load_dataset()
    print("Dataset loaded successfully.")
    
    # Record start time
    start_time = time.time()
    
    # Set up and run the experiment
    manager = ExperimentManager()
    manager.setup_experiments(n_clusters=len(documents.target_names))
    manager.run_all(documents)
    
    # Visual comparison
    manager.visualize_results()
    
    # evaluation indicators comparison
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
        
        for experiment_name, result in manager.results.items():
            f.write(f"\n=== {experiment_name.upper()} ===\n")
            
            # cluster distribution
            clusters = result['clusters']
            unique, counts = np.unique(clusters, return_counts=True)
            f.write("\nCluster Distribution:\n")
            for cluster, count in zip(unique, counts):
                f.write(f"Cluster {cluster}: {count} documents\n")
            
            # Store the main words of each cluster
            f.write("\nTop Terms per Cluster:\n")
            for cluster_id, terms in result['cluster_terms'].items():
                f.write(f"\nCluster {cluster_id}:\n")
                f.write(", ".join(terms[:10]))
                f.write("\n")
        
        # Save execution time
        total_time = time.time() - start_time
        f.write(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print(f"\nResults saved to {results_file}")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
