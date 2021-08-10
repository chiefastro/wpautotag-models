import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from wpautotag_models.nlp.clean import simple_clean


def model_fn(model_dir):
    model = pickle.load(open(os.path.join(model_dir, 'model.pkl'), 'rb'))
    return model

class BlogMultiLabelTopicModel():
    """Recommend topics for blog articles using a supervised multi label model
    trained on the most common tags across domains"""
    def __init__(self, topn=20):
        # Vectorize text
        self.tv = TfidfVectorizer(ngram_range=(1,2), max_features=100000)
        # Convert list of tags per article to multi label target
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        # Fit multi label classifier with logit for each label
        self.clf = LogisticRegression()
        self.multi_clf = MultiOutputClassifier(self.clf, n_jobs=-1)
        self.topn = topn

    def fit(self, X, y):
        # Clean text
        X = pd.Series(X).apply(simple_clean)
        # Vectorize text
        X = self.tv.fit_transform(X)
        # Convert list of tags per article to multi label target
        y = self.mlb.fit_transform(y)
        # Fit multi label classifier with logit for each label
        self.multi_clf.fit(X, y)
        # Normalize predicted probabilities by prevalence of each label
        self.prevalences = y.sum(axis=0) / y.shape[0]

    def predict(self, X):
        # Clean text
        X = pd.Series(X).apply(simple_clean)
        # Vectorize text
        X = self.tv.transform(X)
        # Get predicted probability for each article for each label
        y_pred = np.array(self.multi_clf.predict_proba(X))
        # Convert to df
        y_pred_df = pd.DataFrame(y_pred[:,:,1].T, columns=self.mlb.classes_)
        # Normalize predicted probabilities by prevalence of each label
        y_pred_norm = y_pred_df / self.prevalences
        # Collect top n topics per doc
        return y_pred_norm.apply(self.get_topn, axis=1).tolist()

    def get_topn(self, row):
        """Sort tags by score, filter to self.topn, then collect into list of
        tuples of tag, score"""
        tags = row.T.sort_values(ascending=False).head(self.topn)
        return [(tag, score) for tag, score in zip(tags.index, tags.values)]
