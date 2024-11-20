import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.utils.pipeline import DocumentClusteringPipeline

class ExperimentManager:
    def __init__(self):
        self.pipelines = {}
        self.results = {}
        self.n_clusters = None  
        
    def setup_experiments(self, n_clusters):  
        self.n_clusters = n_clusters  
        combinations = [
             ('tfidf', 'kmeans'),
           # ('tfidf', 'dbscan'),
          # ('fasttext', 'kmeans'),
           # ('fasttext', 'dbscan'),
            #('minilm', 'dbscan'),
            ('minilm', 'kmeans')
           
        ]
        
        for vec_name, clust_name in combinations:
            name = f"{vec_name}_{clust_name}"
            self.pipelines[name] = DocumentClusteringPipeline(
                vectorizer_name=vec_name,  
                clusterer_name=clust_name,  
                n_clusters=n_clusters if clust_name == 'kmeans' else None
            )

    
    def run_all(self, documents):
        for name, pipeline in self.pipelines.items():
            print(f"Running experiment: {name}")
            self.results[name] = pipeline.process(documents)
    
    def visualize_results(self, plot_type='all'):
        """
        可视化实验结果
        
        Args:
            plot_type: 可选 'scatter', 'distribution', 'terms', 'all'
        """
        if plot_type in ['scatter', 'all']:
            self._plot_scatter_clusters()
        
        if plot_type in ['distribution', 'all']:
            self._plot_cluster_distributions()
            
        if plot_type in ['terms', 'all']:
            self._plot_term_importance()
    
    def _plot_scatter_clusters(self):
    
        n_experiments = len(self.results)
        cols = min(3, n_experiments)
        rows = (n_experiments + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flat
        
        for (name, result), ax in zip(self.results.items(), axes):
            vectors_2d = result['reduced_vectors']
            clusters = result['clusters']
            scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                               c=clusters, cmap='viridis')
            ax.set_title(name)
            plt.colorbar(scatter, ax=ax)
            
        
        for ax in axes[len(self.results):]:
            ax.remove()
            
        plt.tight_layout()
        plt.show()
    
    def _plot_cluster_distributions(self):
        
        n_experiments = len(self.results)
        cols = min(3, n_experiments)
        rows = (n_experiments + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flat
        
        for (name, result), ax in zip(self.results.items(), axes):
            clusters = result['clusters']
            unique, counts = np.unique(clusters, return_counts=True)
            ax.bar(unique, counts)
            ax.set_title(f"{name}\nCluster Distribution")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            
        
        for ax in axes[len(self.results):]:
            ax.remove()
            
        plt.tight_layout()
        plt.show()
    
    def _plot_term_importance(self):
        
        for name, result in self.results.items():
            if 'cluster_terms' not in result:
                continue
                
            cluster_terms = result['cluster_terms']
            n_clusters = len(cluster_terms)
            
            fig, axes = plt.subplots(n_clusters, 1, 
                                   figsize=(12, 3*n_clusters))
            if n_clusters == 1:
                axes = [axes]
                
            fig.suptitle(f"{name} - Top Terms by Cluster")
            
            for cluster_id, terms in cluster_terms.items():
                ax = axes[cluster_id]
                top_terms = terms[:10]  
                y_pos = np.arange(len(top_terms))
                
                ax.barh(y_pos, [1]*len(top_terms))
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_terms)
                ax.invert_yaxis()
                ax.set_title(f'Cluster {cluster_id}')
                
            plt.tight_layout()
            plt.show()
    
    def compare_metrics(self):
        """Compare evaluation metrics of different methods"""
        metrics_df = pd.DataFrame({
            name: result['metrics'] 
            for name, result in self.results.items()
        })
        
        # Add a visual display
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar')
        plt.title("Clustering Metrics Comparison")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        return metrics_df
    
    def summarize_results(self):
        """Print summary of experimental results"""
        for name, result in self.results.items():
            print(f"\nExperiment: {name}")
            
            print("\nClustering Metrics:")
            for metric, value in result['metrics'].items():
                print(f"{metric}: {value:.3f}")
                
            print("\nCluster Sizes:")
            unique, counts = np.unique(result['clusters'], return_counts=True)
            for cluster, count in zip(unique, counts):
                print(f"Cluster {cluster}: {count} documents")
            
            if 'cluster_terms' in result:
                print("\nTop Terms per Cluster:")
                for cluster_id, terms in result['cluster_terms'].items():
                    print(f"\nCluster {cluster_id}:")
                    print(", ".join(terms[:10]))