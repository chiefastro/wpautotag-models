import os
import pickle

from bs4 import BeautifulSoup
import yake


def model_fn(model_dir):
    model = pickle.load(open(os.path.join(model_dir, 'model.pkl'), 'rb'))
    return model

class BlogKeyphraseExtractor():
    """Thin wrapper for YAKE!"""
    def __init__(self):
        self.yake_extractor = yake.KeywordExtractor()

    def fit(self, X, y):
        pass

    def predict(self, X):
        """Extract keyphrases

        X is an iterable of raw html docs
        for each doc:
            * parse into plain text with BeautifulSoup
            * apply YAKE! to text
        """
        return [
            self.yake_extractor.extract_keywords(
                BeautifulSoup(html, 'lxml').text
            ) for html in X
        ]
