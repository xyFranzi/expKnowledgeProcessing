import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.utils.pipeline import DocumentClusteringPipeline
import os
import shutil
from datetime import datetime

class ExperimentManager:
    def __init__(self):
        self.pipelines = {}
        self.results = {}
        self.n_clusters = None  
        self.result_folder = None
        
    def setup_experiments(self, n_clusters):  
        self.n_clusters = n_clusters  
        combinations = [
            ('tfidf', 'kmeans'),
            ('tfidf', 'dbscan'),
            ('tfidf', 'hdbscan'),
            ('fasttext', 'kmeans'),
            ('fasttext', 'dbscan'),
            ('fasttext', 'hdbscan'),
            ('minilm', 'kmeans'),
            ('minilm', 'dbscan'),
            ('minilm', 'hdbscan')  
        ]
        
        for vec_name, clust_name in combinations:
            name = f"{vec_name}_{clust_name}"
            self.pipelines[name] = DocumentClusteringPipeline(
                vectorizer_name=vec_name,  
                clusterer_name=clust_name,  
                n_clusters=n_clusters if clust_name == 'kmeans' else None
            )

    def run_all(self, documents):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_folder = os.path.join("results", f"run_{timestamp}")
        os.makedirs(self.result_folder, exist_ok=True)

        for name, pipeline in self.pipelines.items():
            print(f"Running experiment: {name}")
            result = pipeline.process(documents)
            self.results[name] = result
            self.save_text_results(name, result)

    def save_text_results(self, experiment_name, result):
        """Save experimental results as text file"""
        text_file_path = os.path.join(self.result_folder, f"{experiment_name}_results.txt")
        with open(text_file_path, "w", encoding="utf-8") as file:
            file.write(f"Experiment: {experiment_name}\n\n")
            file.write("Clustering Metrics:\n")
            for metric, value in result["metrics"].items():
                file.write(f"{metric}: {value:.3f}\n")
            file.write("\nCluster Sizes:\n")
            unique, counts = np.unique(result["clusters"], return_counts=True)
            for cluster, count in zip(unique, counts):
                file.write(f"Cluster {cluster}: {count} documents\n")

            if "cluster_terms" in result:
                file.write("\nTop Terms per Cluster:\n")
                for cluster_id, terms in result["cluster_terms"].items():
                    file.write(f"\nCluster {cluster_id}:\n")
                    file.write(", ".join(terms[:10]) + "\n")

    def visualize_results(self):
        """Visualize all experimental results in one large image"""
        n_experiments = len(self.results)
        n_cols = 3  
        n_rows = (n_experiments + n_cols - 1) // n_cols
        # 1. Scatter
        plt.figure(figsize=(15, 5*n_rows))
        for idx, (name, result) in enumerate(self.results.items()):
            plt.subplot(n_rows, n_cols, idx + 1)
            vectors_2d = result['reduced_vectors']
            clusters = result['clusters']
            scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            plt.title(f"{name}")
            plt.colorbar(scatter)
        plt.tight_layout()
        scatter_path = os.path.join(self.result_folder, "all_scatter_plots.png")
        plt.savefig(scatter_path)
        plt.close()

        # 2. Cluster distribution
        plt.figure(figsize=(15, 5*n_rows))
        for idx, (name, result) in enumerate(self.results.items()):
            plt.subplot(n_rows, n_cols, idx + 1)
            clusters = result['clusters']
            unique, counts = np.unique(clusters, return_counts=True)
            plt.bar(unique, counts)
            plt.title(f"{name}")
            plt.xlabel("Cluster")
            plt.ylabel("Count")
        plt.tight_layout()
        dist_path = os.path.join(self.result_folder, "all_distributions.png")
        plt.savefig(dist_path)
        plt.close()

        # 3. Summary of term importance
        plt.figure(figsize=(15, 5*n_rows))
        for idx, (name, result) in enumerate(self.results.items()):
            if 'cluster_terms' not in result:
                continue
            plt.subplot(n_rows, n_cols, idx + 1)
            cluster_terms = result['cluster_terms']
            # Show only the first 10 words of the first cluster
            if len(cluster_terms) > 0:
                first_cluster = list(cluster_terms.keys())[0]
                terms = cluster_terms[first_cluster][:10]
                plt.barh(range(len(terms)), [1]*len(terms))
                plt.yticks(range(len(terms)), terms)
                plt.title(f"{name} - Cluster 0")
        plt.tight_layout()
        terms_path = os.path.join(self.result_folder, "all_terms.png")
        plt.savefig(terms_path)
        plt.close()

        # 4. Indicator comparison chart
        metrics_df = pd.DataFrame({
            name: result['metrics']
            for name, result in self.results.items()
        })
        
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar')
        plt.title("Clustering Metrics Comparison")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        metrics_path = os.path.join(self.result_folder, "metrics_comparison.png")
        plt.savefig(metrics_path)
        plt.close()

    def compare_metrics(self):
        """Compare evaluation metrics of different methods and save the comparison plot."""
        metrics_df = pd.DataFrame({
            name: result['metrics']
            for name, result in self.results.items()
        })
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        width = 0.08
        x = np.arange(4)
        methods = metrics_df.columns
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(methods)))
        
        ax2 = ax1.twinx()
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            ax1.bar(x[:3] + i*width, 
                    metrics_df[method][['nmi', 'ari', 'silhouette']], 
                    width, 
                    label=method,
                    color=color)
            
            ax2.bar(x[3] + i*width, 
                    metrics_df[method]['calinski'], 
                    width,
                    color=color)
        
        ax1.set_ylabel('Score (NMI, ARI, Silhouette)')
        ax2.set_ylabel('Score (Calinski-Harabasz)')
        
        ax1.set_ylim(-0.3, 1.0)
        ax2.set_ylim(-50, 250)  
        
        ax1.set_xticks(x + width * len(methods) / 2)
        ax1.set_xticklabels(['NMI', 'ARI', 'Silhouette', 'Calinski'])
        
        plt.title('Clustering Metrics Comparison')
        ax1.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        
        plt.tight_layout()
        
        if self.result_folder:
            csv_path = os.path.join(self.result_folder, "metrics_comparison.csv")
            metrics_df.to_csv(csv_path)
            plot_path = os.path.join(self.result_folder, "metrics_comparison.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Metrics comparison data saved to {csv_path}")
            print(f"Plot saved to {plot_path}")
        
        plt.show()
        return metrics_df

    def summarize_results(self):
        """Print and save a summary of experimental results, including a combined summary file."""
        combined_summary = []  # save all content of exp
        summary_file_path = None
        if self.result_folder:
            summary_file_path = os.path.join(self.result_folder, "experiment_summary.txt")
            combined_summary_file_path = os.path.join(self.result_folder, "combined_experiment_summary.txt")
        
        # traversing every exp generate results seperately
        for name, result in self.results.items():
            summary = []  # each exp content
            
            summary.append(f"Experiment: {name}\n")
            summary.append("\nClustering Metrics:\n")
            for metric, value in result['metrics'].items():
                summary.append(f"{metric}: {value:.3f}\n")
            
            summary.append("\nCluster Sizes:\n")
            unique, counts = np.unique(result['clusters'], return_counts=True)
            for cluster, count in zip(unique, counts):
                summary.append(f"Cluster {cluster}: {count} documents\n")
            
            if 'cluster_terms' in result:
                summary.append("\nTop Terms per Cluster:\n")
                for cluster_id, terms in result['cluster_terms'].items():
                    summary.append(f"\nCluster {cluster_id}:\n")
                    summary.append(", ".join(terms[:10]) + "\n")
        
            # attach to the summary file
            combined_summary.extend(summary)
            combined_summary.append("\n" + "-" * 80 + "\n")
            
            # save each exp file
            if self.result_folder:
                individual_summary_file_path = os.path.join(self.result_folder, f"{name}_summary.txt")
                with open(individual_summary_file_path, "w", encoding="utf-8") as file:
                    file.write("".join(summary))
                print(f"Individual summary saved to {individual_summary_file_path}")
    
        # save summary file
        if summary_file_path:
            with open(combined_summary_file_path, "w", encoding="utf-8") as file:
                file.write("".join(combined_summary))
            print(f"Combined summary saved to {combined_summary_file_path}")

