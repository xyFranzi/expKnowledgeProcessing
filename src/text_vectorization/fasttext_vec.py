import numpy as np
import fasttext
import fasttext.util

class CustomFastTextVectorizer:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def _average_vector(self, document):

        words = document.split()  # Simple whitespace tokenization
        vectors = [self.model.get_word_vector(word) for word in words if word in self.model]
        if not vectors:  # If no words have embeddings
            return np.zeros(self.model.get_dimension())
        return np.mean(vectors, axis=0)

        return np.array([self._average_vector(doc) for doc in documents])

    def transform(self, documents):
        return np.array([self._average_vector(doc) for doc in documents])
