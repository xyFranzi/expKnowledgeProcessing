
# import the dataset from sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns


# Loading 20 Newsgroups dataset
# Use some categories for testing
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space'
]
dataset = fetch_20newsgroups(
    subset='train',  
    categories=categories,
    remove=('headers', 'footers', 'quotes'),  # Remove interference information such as headers
    random_state=42
)