# src/text_vectorization/fasttext_vectorizer.py
import numpy as np
import fasttext
import fasttext.util

class CustomFastTextVectorizer:
    def __init__(self, model_path):
        """
        Initialize the FastText vectorizer.
        
        Args:
            model_path: Path to the FastText pre-trained model file.
        """
        self.model = fasttext.load_model(model_path)

    def _average_vector(self, document):
        """
        Compute the average vector for a given document using FastText embeddings.
        
        Args:
            document: A string containing a document.
            
        Returns:
            Numpy array representing the averaged vector for the document.
        """
        words = document.split()  # Simple whitespace tokenization
        vectors = [self.model.get_word_vector(word) for word in words if word in self.model]
        if not vectors:  # If no words have embeddings
            return np.zeros(self.model.get_dimension())
        return np.mean(vectors, axis=0)

    def fit_transform(self, documents):
        """
        Transform the input documents into vectors using the FastText model.
        
        Args:
            documents: List of text documents
            
        Returns:
            Numpy array of vectors for the documents.
        """
        return np.array([self._average_vector(doc) for doc in documents])

    def transform(self, documents):
        """
        Transform new documents using the FastText model.
        
        Args:
            documents: List of text documents
            
        Returns:
            Numpy array of vectors for the documents.
        """
        return np.array([self._average_vector(doc) for doc in documents])
