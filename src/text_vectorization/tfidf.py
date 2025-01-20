from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

class CustomTfidfVectorizer: 
    def __init__(self, max_features=5000, min_df=5, max_df=0.95):
        self.vectorizer = SklearnTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True
            #token_pattern=r'\b\w+\b' 
        )
        
    def fit_transform(self, documents):
        return self.vectorizer.fit_transform(documents)
    
    def transform(self, documents):
        return self.vectorizer.transform(documents)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()