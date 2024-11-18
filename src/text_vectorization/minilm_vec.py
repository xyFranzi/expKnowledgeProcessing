from sentence_transformers import SentenceTransformer


class CustomMiniLMVectorizer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize MiniLM vectorizer.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model = SentenceTransformer(model_name)
        
    def fit_transform(self, documents):
        """
        Transform documents to embeddings.
        
        Args:
            documents: List of text documents
            
        Returns:
            Document embeddings as numpy array
        """
        return self.model.encode(documents, show_progress_bar=True)
    
    def transform(self, documents):
        """
        Transform new documents to embeddings.
        
        Args:
            documents: List of text documents
            
        Returns:
            Document embeddings as numpy array
        """
        return self.model.encode(documents, show_progress_bar=True)
    