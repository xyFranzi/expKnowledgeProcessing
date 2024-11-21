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

            # self.results[name] = pipeline.process(documents)

    def save_text_results(self, experiment_name, result):
        """将实验结果保存为文本文件"""
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
        """visualise and save experiment results"""
        self._plot_scatter_clusters()
        self._plot_cluster_distributions()
        self._plot_term_importance()
    
    def _plot_scatter_clusters(self):
        for name, result in self.results.items():
            vectors_2d = result['reduced_vectors']
            clusters = result['clusters']

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=clusters, cmap='viridis')
            plt.title(f"{name} - Scatter Plot")
            plt.colorbar(scatter)

            # save img
            image_path = os.path.join(self.result_folder, f"{name}_scatter.png")
            plt.savefig(image_path)
            plt.close()

    def _plot_cluster_distributions(self):
        for name, result in self.results.items():
            clusters = result['clusters']
            unique, counts = np.unique(clusters, return_counts=True)

            plt.figure(figsize=(8, 6))
            plt.bar(unique, counts)
            plt.title(f"{name} - Cluster Distribution")
            plt.xlabel("Cluster")
            plt.ylabel("Count")

            image_path = os.path.join(self.result_folder, f"{name}_distribution.png")
            plt.savefig(image_path)
            plt.close()
    
    def _plot_term_importance(self):
        for name, result in self.results.items():
            if 'cluster_terms' not in result:
                continue

            cluster_terms = result['cluster_terms']
            n_clusters = len(cluster_terms)

            plt.figure(figsize=(12, 6))
            for cluster_id, terms in cluster_terms.items():
                plt.barh(terms[:10], range(10))
                plt.title(f"{name} - Cluster {cluster_id} Top Terms")

            image_path = os.path.join(self.result_folder, f"{name}_terms.png")
            plt.savefig(image_path)
            plt.close()

    def compare_metrics(self):
        """Compare evaluation metrics of different methods and save the comparison plot."""
        metrics_df = pd.DataFrame({
            name: result['metrics']
            for name, result in self.results.items()
        })
        
        # Visualize metrics
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar')
        plt.title("Clustering Metrics Comparison")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the comparison plot
        if self.result_folder:
            comparison_plot_path = os.path.join(self.result_folder, "metrics_comparison.png")
            plt.savefig(comparison_plot_path)
            print(f"Metrics comparison plot saved to {comparison_plot_path}")
        
        plt.show()
        
        # Save the metrics dataframe as a CSV file
        if self.result_folder:
            csv_path = os.path.join(self.result_folder, "metrics_comparison.csv")
            metrics_df.to_csv(csv_path)
            print(f"Metrics comparison data saved to {csv_path}")
        
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
