from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

class CustomTfidfVectorizer:  # 改名为CustomTfidfVectorizer
    def __init__(self, max_features=5000, min_df=5, max_df=0.95):
        """
        Initialize TF-IDF vectorizer with custom parameters.
        
        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.vectorizer = SklearnTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True
            #token_pattern=r'\b\w+\b'  # If you need to customize the word mode
        )
        
    def fit_transform(self, documents):
        """
        Fit the vectorizer and transform the documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            sparse matrix of TF-IDF features
        """
        return self.vectorizer.fit_transform(documents)
    
    def transform(self, documents):
        """
        Transform documents using the fitted vectorizer.
        
        Args:
            documents: List of text documents
            
        Returns:
            sparse matrix of TF-IDF features
        """
        return self.vectorizer.transform(documents)
    
    def get_feature_names(self):
        """
        Get feature names (vocabulary).
        
        Returns:
            list of feature names
        """
        return self.vectorizer.get_feature_names_out()