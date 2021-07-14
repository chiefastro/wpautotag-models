import os
import pickle

from bs4 import BeautifulSoup
import yake


def model_fn(model_dir):
    model = pickle.load(open(os.path.join(model_dir, 'model.pkl'), 'rb'))
    return model

class BlogMultiLabelTopicModel():
    """Recommend topics for blog articles using a supervised multi label model
    trained on the most common tags across domains"""
    def __init__(self):
        # Vectorize text
        self.tv = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
        # Convert list of tags per article to multi label target
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        # Fit multi label classifier with logit for each label
        self.clf = LogisticRegression()
        self.multi_clf = MultiOutputClassifier(self.clf, n_jobs=-1)

    def fit(self, X, y):
        # Vectorize text
        X = self.tv.fit_transform(X)
        # Convert list of tags per article to multi label target
        y = self.mlb.fit_transform(y)
        # Fit multi label classifier with logit for each label
        self.multi_clf.fit(X, y)
        # Normalize predicted probabilities by prevalence of each label
        self.prevalences = y.sum(axis=0) / y.shape[0]

    def predict(self, X):
        # Vectorize text
        X = self.tv.transform(X)
        # Get predicted probability for each article for each label
        y_pred = self.multi_clf.predict_proba(X)
        # Normalize predicted probabilities by prevalence of each label
        y_pred_norm = y_pred / prevalences
        return y_pred_norm
