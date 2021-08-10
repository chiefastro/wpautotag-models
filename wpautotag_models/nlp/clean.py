"""Module for text cleaning"""

import re

def simple_clean(text):
    text = str(text)
    text = text.lower()
    # remove html tags
    text = re.sub('<.*?>', '', text)
    # remove urls within text
    text = re.sub('http.+[\s\\n]', '', text)
    # keep alphabetic only
    text = re.sub('[\W_]+', ' ', text)
    # trim whitespace
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def sluglike(text):
    # lowercase
    text = text.lower()
    # keep alphanumeric only
    text = re.sub('[\W_]+', ' ', text)
    # remove duplicate spaces
    text = re.sub(' +', ' ', text)
    # strip leading and trailing whitespace
    text = text.strip()
    return text
