import pandas as pd
import numpy as np
import os
import re
import pickle
import math
from collections import defaultdict

from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

WORD_VECT_DIR = os.path.join('D:', 'data_science', 'common')
WORD_VECT_FPATHS = {
    'word2vec': os.path.join(
        WORD_VECT_DIR, 'GoogleNews-vectors-negative300.bin'
    ),
    'common_crawl': os.path.join(WORD_VECT_DIR, 'crawl-300d-2M.vec'),
    'glove': os.path.join(WORD_VECT_DIR, 'glove.840B.300d.txt'),
    'paragram': os.path.join(WORD_VECT_DIR, 'paragram_300_sl999.txt'),
    'wiki_news': os.path.join(WORD_VECT_DIR, 'wiki-news-300d-1M.vec'),
}
WORD_VECT_COLS = list(WORD_VECT_FPATHS.keys())


def model_fn(model_dir):
    model = pickle.load(open(os.path.join(model_dir, 'model.pkl'), 'rb'))
    return model


class TextPriorTransformer():
    def __init__(
        self,
        cache_dir=None,
        update_cache=False,
        domain_freq_thresh=0.0015,
        cats_drop_post_canon_fit=['blog', 'other', 'general', 'tips']
    ):
        self.cache_dir = cache_dir
        self.update_cache = update_cache
        self.domain_freq_thresh = domain_freq_thresh
        self.cats_drop_post_canon_fit = cats_drop_post_canon_fit

    def fit(self, X, y, groups):
        # split text and prior from X
        df = pd.DataFrame(X)
        text = df['text']
        prior = df['prior']
        category = y.apply(sluglike)

        print('before fit canonical map')
        # fit canonical_map
        self.fit_canonical_map(text, category, groups)

        print('before prior category clean')
        # clean category names in prior
        prior_clean = {sluglike(cat): v for cat, v in prior}
        print('before canonicalize prior')
        # get canonical prior plus norm
        prior_canonical = self.canonicalize_prior(prior_clean)
        print('before normalize prior')
        prior_canonical_norm = self.normalize_prior(prior_canonical)

        print('before text clean')
        # clean text
        text_clean = text.apply(simple_clean)
        print('before count tokens')
        # count tokens
        num_tokens = text_clean.apply(
            lambda x: len(x.split())
        ).values.reshape(-1, 1)
        print('before tfidf')
        # tfidf
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', max_features=10000,
            use_idf=True, binary=False
        )
        self.tfidf_vectorizer.fit(text_clean)
        tfidf_vect = self.tfidf_vectorizer.transform(text_clean)
        print('before stack')
        # stack
        print('tfidf', tfidf_vect.shape)
        print('prior_canonical', prior_canonical.shape)
        print('prior_canonical_norm', prior_canonical_norm.shape)
        print('num_tokens', num_tokens.shape)
        X = scipy.sparse.hstack(
            [
                tfidf_vect,
                prior_canonical,
                prior_canonical_norm,
                num_tokens
            ]
        )
        print('before scale')
        # scale
        self.scaler = MaxAbsScaler()
        self.scaler.fit(X)

    def fit_canonical_map(self, text, category, groups):
        # get category domain counts
        groups_df = pd.DataFrame({'category': category, 'domain': groups})
        category_domain_counts = pd.Series(
            groups_df.groupby('domain')['category']\
            .apply(set).apply(list).sum()
        ).value_counts()

        # get similarity scores for category pairs
        scores_df = get_similarity_scores(
            text, category,
            category_domain_counts,
            domain_freq_thresh=self.domain_freq_thresh,
            cache_dir=self.cache_dir, update_cache=self.update_cache
        )

        # apply classifier (heuristics)
        scores_df['link_score'] = get_link_scores(
            scores_df, domain_freq_thresh=self.domain_freq_thresh
        )

        # get map from category to canonical category
        self.canonical_map_ = get_canonical_map(
            scores_df, category_domain_counts,
            domain_freq_thresh=self.domain_freq_thresh
        )

        # set consistent order of categories for prior vectors
        self.category_order = np.unique(
            list(self.canonical_map_.values())
        ).tolist()

        # set reverse canonical map for fast lookups of equivalent_categories
        self.set_reverse_canonical_map(category_domain_counts)

    def set_reverse_canonical_map(self, category_domain_counts):
        # set reverse canonical map for fast lookups of equivalent_categories
        reverse_canonical_map = defaultdict(list)
        for cat, canon_cat in self.canonical_map_.items():
            reverse_canonical_map[canon_cat] += [cat]
        # order equivalent_categories by domain count descending
        reverse_canonical_map_ordered = {}
        def count_lookup(x):
            try:
                return category_domain_counts[x]
            except:
                return 0
        for canon_cat, equivalent_cats in reverse_canonical_map.items():
            ordered_cats = sorted(
                equivalent_cats,
                key=count_lookup,
                reverse=True
            )
            # exclude canon_cat if
            try:
                ordered_cats.remove(canon_cat)
            except:
                pass
            reverse_canonical_map_ordered[canon_cat] = ordered_cats
        self.reverse_canonical_map_ = reverse_canonical_map_ordered



    def canonicalize_prior(self, prior):
        canonical_prior_xcoords = []
        canonical_prior_ycoords = []
        canonical_prior_values = []
        for i, this_prior in enumerate(prior):
            # aggregate prior counts to canonical category
            this_canonical_prior = defaultdict(int)
            for cat, count in this_prior.items():
                canon_cat = self.canonical_map_[cat] if (
                    cat in self.canonical_map_) else 'other'
                this_canonical_prior[canon_cat] += count
            # consistently order and drop unusable cats
            for cat, count in this_canonical_prior.items():
                if cat not in self.cats_drop_post_canon_fit:
                    canonical_prior_xcoords.append(i)
                    canonical_prior_ycoords.append(
                        self.category_order.index(cat)
                    )
                    canonical_prior_values.append(count)
        # sparse csr from coords
        val_coords_tup = (
            canonical_prior_values,
            (canonical_prior_xcoords, canonical_prior_ycoords)
        )
        canonical_prior = scipy.sparse.csr_matrix(
            val_coords_tup,
            shape=(len(prior), len(self.category_order))
        )
        return canonical_prior

    def normalize_prior(self, prior):
        # divide by sum
        prior_norm = prior / prior.sum(axis=1)
        # fill na with 0
        prior_norm = np.nan_to_num(prior_norm, nan=0)
        return prior_norm

    def transform(self, X):
        # split text and prior from X
        df = pd.DataFrame(X)
        text = df['text']
        prior = df['prior']

        # get canonical prior plus norm
        prior_clean = {sluglike(cat): v for cat, v in prior}
        prior_canonical = self.canonicalize_prior(prior_clean)
        prior_canonical_norm = self.normalize_prior(prior_canonical)

        # clean text
        text_clean = text.apply(simple_clean)
        # count tokens
        num_tokens = text_clean.apply(
            lambda x: len(x.split())
        ).values.reshape(-1, 1)
        # tfidf
        tfidf_vect = self.tfidf_vectorizer.transform(text_clean)
        # stack
        X = scipy.sparse.hstack(
            [
                tfidf_vect,
                prior_canonical,
                prior_canonical_norm,
                num_tokens
            ]
        )
        # scale
        X = self.scaler.transform(X)

        return X


class BlogCategoryModel():
    def __init__(
        self,
        cache_dir=None,
        update_cache=False,
        domain_freq_thresh=0.0015,
        random_state=12345,
        cats_drop_pre_canon_fit=['uncategorized', 'uncategorised'],
        cats_drop_post_canon_fit=['blog', 'other', 'general', 'tips'],
    ):
        self.cache_dir = cache_dir
        self.update_cache = update_cache
        self.domain_freq_thresh = domain_freq_thresh
        self.random_state = random_state
        self.cats_drop_pre_canon_fit = cats_drop_pre_canon_fit
        self.cats_drop_post_canon_fit = cats_drop_post_canon_fit

    def fit(self, X, y, groups):
        """"""
        print('before drop pre')
        # drop uncategorized
        keep_mask = list(~y.isin(self.cats_drop_pre_canon_fit))
        # have to keep X as a list
        X = [x for i, x in enumerate(X) if keep_mask[i]]
        y = y[keep_mask].copy()

        print('before transform')
        # transform X
        self.tpt = TextPriorTransformer(
            cache_dir=self.cache_dir,
            update_cache=self.update_cache,
            domain_freq_thresh=self.domain_freq_thresh,
            cats_drop_post_canon_fit=self.cats_drop_post_canon_fit
        )
        self.tpt.fit(X, y, groups)
        X = self.tpt.transform(X)

        # give access to canonical map and it's reverse
        self.canonical_map_ = self.tpt.canonical_map_.copy()
        self.reverse_canonical_map_ = self.tpt.reverse_canonical_map_.copy()
        print('after transform')

        # transform y with canonical map
        y = self.canonicalize(y)

        print('before drop post')
        # drop unusable categories
        keep_mask = ~y.isin(self.cats_drop_post_canon_fit)
        keep_mask_numpy = keep_mask.astype(int).values.nonzero()[0]
        X = X[keep_mask_numpy,:].copy()
        y = y[keep_mask].copy()

        # encode labels
        self.le = LabelEncoder()
        self.le.fit(y)
        y = self.le.transform(y)

        print('before model fit')
        # fit model
        self.model = LogisticRegression(
            C=1, # default but also best from random search
            solver='lbfgs', random_state=12345, penalty='l2',
            fit_intercept=True, max_iter=3000, multi_class='ovr'
        )
        self.model.fit(X, y)
        print('after model fit')

    def predict(self, X):
        # transform X
        X = self.tpt.transform(X)
        # predict y
        y_pred = self.model.predict(X)
        # get confidence
        confidence = self.model.predict_proba(X).max(axis=1)
        # decode y prediction
        predicted_category = self.le.inverse_transform(y_pred)
        # look up equivalent_categories from canonical_map
        equivalent_categories = [
            self.reverse_canonical_map_[cat] for cat in predicted_category
        ]
        # format for return
        payload = []
        for pred, equiv, conf in zip(
            predicted_category, equivalent_categories, confidence
        ):
            payload.append(
                {
                    'predicted_category': pred,
                    # 'equivalent_categories': equiv,
                    'confidence': conf
                }
            )

        return payload

    def canonicalize(self, y):
        y = y.map(self.canonical_map_)
        y = y.fillna('other')
        return y


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


def get_word_vects(vocab, vect_fpath, file_type='txt'):
    word_vects = {}
    i = 0
    if file_type == 'word2vec':
        word2vec = KeyedVectors.load_word2vec_format(vect_fpath, binary=True)
        for word in vocab:
            if word in word2vec:
                word_vects[word] = word2vec.get_vector(word)
            else:
                word_vects[word] = np.zeros(300)
    elif file_type == 'txt':
        for line in open(vect_fpath, encoding='utf-8', errors='ignore'):
            split = line.split()
            if len(split) > 301:
                # ignore lines with phrases instead of single words
                continue
            if split[0] in vocab:
                try:
                    word_vects[split[0]] = np.array(split[1:], dtype='float32')
                except:
                    continue
        for word in vocab:
            if word not in word_vects:
                word_vects[word] = np.zeros(300)
    return word_vects


def get_all_word_vects(category):
    cat_vocab = set(category.drop_duplicates().apply(
        lambda x: str(x).split()
    ).sum())

    all_word_vects = {}
    for wv_name, wv_fpath in WORD_VECT_FPATHS.items():
        if wv_name == 'word2vec':
            file_type='word2vec'
        else:
            file_type='txt'
        all_word_vects[wv_name] = get_word_vects(
            cat_vocab, wv_fpath, file_type=file_type
        )

    return all_word_vects


def get_tfidf_vects(text, category):
    cats = category.drop_duplicates().tolist()

    # vectorize terms
    vect = TfidfVectorizer(
        stop_words='english', max_features=10000, use_idf=True, binary=False
    )
    vect.fit(text)
    tfidf_mat_articles = vect.transform(text)

    # collect dict keyed on category containing tfidf vectors
    # averaged across documents within category
    tfidf_vects = {}
    for cat in cats:
        mask = (category == cat).astype(int).values.nonzero()[0]
        tfidf_vects[cat] = tfidf_mat_articles[mask].mean(axis=0)

    return tfidf_vects


def get_word_vect(string, word_vects):
    return np.vstack([word_vects[word] for word in string.split()]).mean(axis=0)


def get_word_similarity(string1, string2, word_vects):
    return 1 - scipy.spatial.distance.cosine(
        get_word_vect(string1, word_vects), get_word_vect(string2, word_vects)
    )


def get_tfidf_similarity(string1, string2, tfidf_vects):
    return 1 - scipy.spatial.distance.cosine(
        tfidf_vects[string1], tfidf_vects[string2]
    )


def get_scores_df(
    i, category_domain_counts, min_domain_count, all_word_vects, tfidf_vects
):
    cat_i = category_domain_counts.index[i]
    num_domains_i = category_domain_counts.iloc[i]
    # only consider cat_js with > 1 domains
    cat_js = category_domain_counts[
        category_domain_counts >= min_domain_count
    ].index[:i]
    num_domains_js = category_domain_counts[
        category_domain_counts >= min_domain_count
    ].iloc[:i].values
    # compute similarity scores to every pair of categories
    scores = {'category_j': cat_js, 'num_domains_j': num_domains_js}
    scores['fuzzy_ratio'] = np.array(
        [fuzz.ratio(cat_i, cat_j) / 100. for cat_j in cat_js]
    )
    scores['fuzzy_partial_ratio'] = np.array(
        [fuzz.partial_ratio(cat_i, cat_j) / 100. for cat_j in cat_js]
    )
    scores['fuzzy_token_sort'] = np.array(
        [fuzz.token_sort_ratio(cat_i, cat_j) / 100. for cat_j in cat_js]
    )
    scores['fuzzy_token_set'] = np.array(
        [fuzz.token_set_ratio(cat_i, cat_j) / 100. for cat_j in cat_js]
    )
    scores['tfidf'] = np.array(
        [get_tfidf_similarity(cat_i, cat_j, tfidf_vects) for cat_j in cat_js]
    )
    for wv_name, word_vects in all_word_vects.items():
        scores[wv_name] = np.array(
            [get_word_similarity(cat_i, cat_j, word_vects) for cat_j in cat_js]
        )
    scores_df = pd.DataFrame(scores)
    scores_df['category_i'] = cat_i
    scores_df['num_domains_i'] = num_domains_i
    return scores_df


def get_similarity_scores(
    text, category, category_domain_counts, domain_freq_thresh=0.0005,
    cache_dir=None, update_cache=False
):
    # get similarity scores for category pairs

    # load from cache if available
    if cache_dir is not None:
        cache_fpath = os.path.join(cache_dir, 'category_similarity.pkl')
        if os.path.exists(cache_fpath) and not update_cache:
            return pd.read_pickle(cache_fpath)

    # get all_word_vects
    all_word_vects = get_all_word_vects(category)

    # get tfidf_vects
    tfidf_vects = get_tfidf_vects(text, category)

    # calculate min_domain_count
    num_domains = category_domain_counts.shape[0]
    min_domain_count = math.ceil(num_domains * domain_freq_thresh)

    # calculate scores
    scores_df_list = []
    for i in range(1, category_domain_counts.shape[0]):
        scores_df_list.append(get_scores_df(
            i, category_domain_counts, min_domain_count,
            all_word_vects, tfidf_vects
        ))
    scores_df = pd.concat(scores_df_list)

    # cache
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        scores_df.to_pickle(cache_fpath)

    return scores_df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def zscore(x):
    return (x - x.mean()) / x.std()


def get_link_scores(scores_df, domain_freq_thresh=0.0005):
    # num_domains_correction is a value from 0 to 1,
    # where 0 is domain_freq_thresh_log and 1 is maximum of num_domains,
    # and the scale is sqrt of log
    domain_freq_thresh_log = np.log(domain_freq_thresh)
    num_domains_log_clip = (np.maximum(
        domain_freq_thresh_log,
        np.log(scores_df['num_domains_i'] / scores_df['category_i'].nunique())
    ) - domain_freq_thresh_log)
    num_domains_correction = np.sqrt(
        num_domains_log_clip / num_domains_log_clip.max()
    )

    # fuzzy features
    fuzzy_ratio_z = np.maximum(
        scores_df['fuzzy_ratio'] - 0.8, 0
    ) / scores_df['fuzzy_ratio'].std()
    fuzzy_partial_ratio_z = np.maximum(
        scores_df['fuzzy_partial_ratio'] - 0.9, 0
    ) / scores_df['fuzzy_partial_ratio'].std()
    fuzzy_token_set_z = np.maximum(
        scores_df['fuzzy_token_set'] - 0.8, 0
    ) / scores_df['fuzzy_token_set'].std()
    fuzzy_token_sort_z = np.maximum(
        scores_df['fuzzy_token_sort'] - 0.8, 0
    ) / scores_df['fuzzy_token_sort'].std()

    # tfidf feature, scaled down as num_domains decreases
    tfidf_z = (scores_df['tfidf'] - 0.6) / scores_df['tfidf'].std() * \
        num_domains_correction

    # word vect scores - consider each individually as well as mean and median
    # mean
    word_vect_features = [scores_df[WORD_VECT_COLS].apply(
        lambda x: zscore(x) - 1, axis=0
    ).mean(axis=1)]
    # median
    word_vect_features += [scores_df[WORD_VECT_COLS].apply(
        lambda x: zscore(x) - 1, axis=0
    ).median(axis=1)]
    for wv_name in WORD_VECT_COLS:
        word_vect_features.append((zscore(scores_df[wv_name]) - 1))

    # use num_domains as a feature
    num_domains_feature = (1 - np.power(num_domains_correction, 0.1)) * 5

    # combine z values with a heuristic to generate score
    const = 1 / (6. + len(word_vect_features))
    presig = tfidf_z * const * 2 + \
        fuzzy_ratio_z * const / 2 + \
        fuzzy_partial_ratio_z * const / 2 + \
        fuzzy_token_set_z * const / 2 + \
        fuzzy_token_sort_z * const / 2 + \
        num_domains_feature * const
    for wv_feature in word_vect_features:
        presig += wv_feature * const

    return sigmoid(presig)


def get_canonical_map(
    scores_df, category_domain_counts, domain_freq_thresh=0.0005
):
    # collapse each category to most similar category that is more popular
    # than self and greater than some similarity threshold
    simi_thresh = 0.7
    score_col = 'link_score'
    cat_groups = scores_df.groupby('category_i')
    cat_is = scores_df['category_i'].drop_duplicates().tolist()
    num_domains = category_domain_counts.shape[0]
    i = 0
    score_df_list = []
    # seed with top category
    canonical_map = {
        category_domain_counts.index[0]: category_domain_counts.index[0]
    }
    num_can_cats = 1
    is_can = False
    for cat in cat_is:
        # filter scores to this cat
        scores_df_this = cat_groups.get_group(cat).copy()
        # sort by link_score
        scores_df_this = scores_df_this.sort_values(
            score_col, ascending=False
        ).copy()
        # get closest cat
        closest_cat_row = scores_df_this.head(1)
        closest_cat = closest_cat_row['category_j'].values[0]
        # if max similarity greater than thresh, map this cat to an
        # existing canonical category
        if closest_cat_row[score_col].values[0] > simi_thresh:
            is_can = False
            # map this cat to closest_cat's canonical cat
            canonical_map[cat] = canonical_map[closest_cat]
        # if cat only occurs in one domain, mark as 'other'
        elif (
            closest_cat_row['num_domains_i'].values[0] / num_domains
        ) < domain_freq_thresh:
            is_can = False
            canonical_map[cat] = 'other'
        # if max similarity less than thresh and this cat occurs in
        # more than one domain, this cat is canonical
        else:
            is_can = True
            canonical_map[cat] = cat
            num_can_cats += 1
        i += 1

    return canonical_map
