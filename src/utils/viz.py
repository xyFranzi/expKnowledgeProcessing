from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ClusterVisualizer:
    def __init__(self):
        """Initialize the visualizer"""
        self.svd = TruncatedSVD(n_components=2, random_state=42)
        
    def plot_clusters_2d(self, X, clusters, title="Document Clusters Visualization"):
        """
        Plot clusters in 2D using SVD for dimensionality reduction.
        
        Args:
            X: Document-term matrix
            clusters: Cluster labels
            title: Plot title
        """
        # Reduce dimensionality
        X_2d = self.svd.fit_transform(X)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('First SVD component')
        plt.ylabel('Second SVD component')
        plt.show()
        
    def plot_cluster_distribution(self, clusters):
        """
        Plot the distribution of documents across clusters.
        
        Args:
            clusters: Cluster labels
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=clusters)
        plt.title('Distribution of Documents Across Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Documents')
        plt.show()
        
    def plot_term_importance(self, cluster_terms, top_n=10):
        """
        Plot top terms for each cluster.
        
        Args:
            cluster_terms: Dictionary mapping cluster IDs to lists of terms
            top_n: Number of top terms to show
        """
        n_clusters = len(cluster_terms)
        fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4*n_clusters))
        
        for i, (cluster_id, terms) in enumerate(cluster_terms.items()):
            terms = terms[:top_n]
            ax = axes[i] if n_clusters > 1 else axes
            y_pos = np.arange(len(terms))
            
            ax.barh(y_pos, [1]*len(terms))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(terms)
            ax.invert_yaxis()
            ax.set_title(f'Top Terms in Cluster {cluster_id}')
            
        plt.tight_layout()
        plt.show()

def visualize_clustering_results(X, clusters, cluster_terms):
    """
    Comprehensive visualization of clustering results.
    
    Args:
        X: Document-term matrix
        clusters: Cluster labels
        cluster_terms: Dictionary mapping cluster IDs to lists of terms
    """
    visualizer = ClusterVisualizer()
    
    # Plot clusters in 2D
    visualizer.plot_clusters_2d(X, clusters)
    
    # Plot distribution of documents across clusters
    visualizer.plot_cluster_distribution(clusters)
    
    # Plot top terms for each cluster
    visualizer.plot_term_importance(cluster_terms)