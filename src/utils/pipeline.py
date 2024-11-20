from src.text_vectorization.tfidf import CustomTfidfVectorizer
from src.text_vectorization.fasttext_vec import CustomFastTextVectorizer
from src.text_vectorization.minilm_vec import CustomMiniLMVectorizer
from src.clustering.kmeans import DocumentKMeans
from src.clustering.dbscan import DocumentDBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os

class DocumentClusteringPipeline:
    def __init__(self, vectorizer_name, clusterer_name, n_clusters=None):
        self.vectorizer_name = vectorizer_name
        self.clusterer_name = clusterer_name
        self.n_clusters = n_clusters
        self.results = {}  # 存储中间结果
        self.pca = PCA(n_components=2)  # 添加PCA用于降维
        self.setup_components()
        
    def setup_components(self):
        # 设置向量化方法
        if self.vectorizer_name == 'tfidf':
            self.vectorizer = CustomTfidfVectorizer()
        elif self.vectorizer_name == 'fasttext':
            self.vectorizer = CustomFastTextVectorizer(
                    # model_path='/Users/yue/Documents/code/expKnowledgeProcessing/models/cc.en.300.bin'
                model_path='D:/mypython/KP/expKnowledgeProcessing/models/cc.en.300.bin'
            )
        elif self.vectorizer_name == 'minilm':
            self.vectorizer = CustomMiniLMVectorizer()
        
        # 设置聚类方法
        if self.clusterer_name == 'kmeans':
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for KMeans clustering")
            self.clusterer = DocumentKMeans(n_clusters=self.n_clusters)
        elif self.clusterer_name == 'dbscan':
            self.clusterer = DocumentDBSCAN()
    
    def process(self, documents):
        # 存储原始数据集
        self.results['original_dataset'] = documents
        
        # 向量化
        #  Load previous vectorization file (if there is)
                
        if self.vectorizer_name == 'fasttext':
            if os.path.exists('vectorized_data.pkl'):
                with open('vectorized_data.pkl', 'rb') as f:
                    vectors = pickle.load(f)
                print("Loaded vectorized data from file.")
            else:
            # Perform vectorization as before
                print("Vectorizing dataset...")
                vectors = self.vectorizer.fit_transform(documents.data)
                with open('vectorized_data.pkl', 'wb') as f:
                    pickle.dump(vectors, f)
                print("Vectorization complete and data saved.")
        else:
            vectors = self.vectorizer.fit_transform(documents.data)
            
        self.results['vectors'] = vectors

        # 降维用于可视化
        # 检查vectors是否为稀疏矩阵，如果是则转换为密集矩阵
        if hasattr(vectors, 'toarray'):
            dense_vectors = vectors.toarray()
        else:
            dense_vectors = vectors
        
        reduced_vectors = self.pca.fit_transform(dense_vectors)
        self.results['reduced_vectors'] = reduced_vectors

        # 聚类
        clusters = self.clusterer.fit_predict(vectors)
        self.results['clusters'] = clusters
        
        # 评估
        metrics = self.evaluate(clusters)
        self.results['metrics'] = metrics
        
        return self.results

    def evaluate(self, clusters):
        """
        评估聚类结果，包括内部和外部评估指标
        
        Returns:
            dict containing various evaluation metrics
        """
        true_labels = self.results['original_dataset'].target
        vectors = self.results['vectors']
        
        # 确保向量是numpy数组格式
        if not isinstance(vectors, np.ndarray):
            vectors = vectors.toarray()
            
        metrics = {
            # 外部评估指标（需要真实标签）
            'nmi': normalized_mutual_info_score(true_labels, clusters),
            'ari': adjusted_rand_score(true_labels, clusters),
            
            # 内部评估指标（不需要真实标签）
            'silhouette': silhouette_score(vectors, clusters),
            'calinski': calinski_harabasz_score(vectors, clusters)
        }
        
        # 分析聚类结果
        self.results['cluster_terms'] = self.get_cluster_terms_embeddings(
            self.results['original_dataset'].data,
            clusters
        )
        
        # 找出代表性文档
        self.results['representative_docs'] = self.find_representative_documents(
            vectors,
            clusters,
            self.results['original_dataset'].data
        )
        
        return metrics

    def get_cluster_terms_embeddings(self, texts, clusters, top_n=10, max_features=1000):
        """
        提取每个聚类的主要词语
        
        Args:
            texts: 原始文本列表
            clusters: 聚类标签
            top_n: 每个聚类返回的词语数量
            max_features: 考虑的最大词语数量
        
        Returns:
            Dictionary：{cluster_id: [term1, term2, ...]}
        """ 
        count_vec = CountVectorizer(max_features=max_features, stop_words='english')
        X_count = count_vec.fit_transform(texts)
        feature_names = count_vec.get_feature_names_out()
        
        cluster_terms = {}
        
        for cluster_id in np.unique(clusters):
            cluster_docs = X_count[clusters == cluster_id]
            avg_counts = cluster_docs.mean(axis=0).A1
            top_indices = avg_counts.argsort()[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            cluster_terms[cluster_id] = top_terms
        
        return cluster_terms

    def find_representative_documents(self, X, clusters, documents, n_docs=5):
        """
        找出每个聚类中最具代表性的文档
        
        Args:
            X: 文档的向量表示
            clusters: 聚类标签
            documents: 原始文档列表
            n_docs: 每个聚类返回的文档数量
        
        Returns:
            Dictionary：{cluster_id: [doc1, doc2, ...]}
        """
        unique_clusters = np.unique(clusters)
        representatives = {}
        
        for cluster_id in unique_clusters:
            # 获取该聚类的文档
            cluster_mask = clusters == cluster_id
            cluster_embeddings = X[cluster_mask]
            cluster_docs = np.array(documents)[cluster_mask]
            
            # 计算聚类中心
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # 计算相似度
            similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1))
            
            # 获取最相似的文档
            top_indices = similarities.flatten().argsort()[-n_docs:][::-1]
            representatives[cluster_id] = cluster_docs[top_indices].tolist()
        
        return representatives

    def analyze_clusters(self, kmeans, feature_names, n_terms=10):
        """
        分析每个聚类的主要特征词（仅适用于K-means聚类）
        
        Args:
            kmeans: 已训练的KMeans模型
            feature_names: 特征名称列表
            n_terms: 返回的词语数量
        
        Returns:
            Dictionary：{cluster_id: [term1, term2, ...]}
        """
        cluster_terms = {}
        centroids = kmeans.cluster_centers_
        
        for i in range(len(centroids)):
            top_indices = centroids[i].argsort()[::-1][:n_terms]
            top_terms = [feature_names[idx] for idx in top_indices]
            cluster_terms[i] = top_terms
            
        return cluster_terms