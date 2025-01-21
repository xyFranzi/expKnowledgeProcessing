from sklearn.datasets import fetch_20newsgroups

def load_dataset(categories=None, subset='train'):
    """
    Load the 20 newsgroups dataset.
    
    Args:
        categories: List of categories to load. If None, load default categories
        subset: 'train', 'test' or 'all'
    
    Returns:
        dataset object containing the texts and their labels
    """
    if categories is None:
        # choose 4 categories
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space'
        ]
    dataset = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    return dataset
