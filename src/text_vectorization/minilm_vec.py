from sentence_transformers import SentenceTransformer


class CustomMiniLMVectorizer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def fit_transform(self, documents):
        return self.model.encode(documents, show_progress_bar=True)
    
    def transform(self, documents):
        return self.model.encode(documents, show_progress_bar=True)
    