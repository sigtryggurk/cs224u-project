import numpy as np

from config import Config
from model_utils import dummy_tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class DenseTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit(self, X, y=None, **fit_params):
        return self

class Reshape(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        data = np.array(data)
        assert len(data.shape) == 1
        return data.reshape((-1, 1))

class SklearnModel(object):
    def __init__(self, name, pipe, params_range):
        self.name = name
        self.pipe = pipe
        self.params_range = params_range

def text_selector_pipe(key):
    return Pipeline([
             ('select', ItemSelector(key=key)),
             ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
             ('tfidf', TfidfTransformer())
             ])

def text_pipe(clf):
    return Pipeline([
             ('select', ItemSelector(key="question")),
             ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
             ('tfidf', TfidfTransformer()),
             ('clf', clf)
             ])

def text_dense_pipe(clf):
    return Pipeline([
             ('select', ItemSelector(key="question")),
             ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
             ('tfidf', TfidfTransformer()),
             ('dense', DenseTransformer()),
             ('clf', clf)
             ])

def text_and_scalar_pipe(scalar, clf):
    return Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('text', text_selector_pipe('question')),
                ('scalar', Pipeline([
                    ('select', ItemSelector(key=scalar)),
                    ('reshape', Reshape()),
                    ]))
                ]
            )),
        ('clf', clf)
        ])

def multi_text_pipe(texts, clf):
    transformer_list = [('question', text_selector_pipe('question'))]
    for text in texts:
        transformer_list.append((text, text_selector_pipe(text)))
    return Pipeline([
        ('union', FeatureUnion(
            transformer_list = transformer_list
            )),
        ('clf', clf)
        ])

log_params = {'clf__C': np.logspace(-1,4,100), 'clf__penalty': ['l2', 'l1']}
svm_params = {'clf__C': np.logspace(-1,4,100), 'clf__loss': ['squared_hinge']}
rf_params = {'clf__n_estimators': range(5,20), 'clf__criterion': ['gini', 'entropy'], 'clf__max_features':['auto', 'sqrt','log2', None], 'clf__n_jobs':[4]}

Logistic = SklearnModel("logistic", text_pipe(LogisticRegression(class_weight='balanced', random_state=Config.SEED)), log_params)
SVM = SklearnModel("svm", text_pipe(LinearSVC(class_weight='balanced', random_state=Config.SEED)), svm_params)
Dummy = SklearnModel("dummy", text_pipe(DummyClassifier(random_state=Config.SEED)), {})
RandomForest = SklearnModel("random_forest", text_dense_pipe(RandomForestClassifier(random_state=Config.SEED)), rf_params)
NB = SklearnModel("nb", text_dense_pipe(GaussianNB()), {})

LogisticWithScalar = lambda s: SklearnModel("logistic", text_and_scalar_pipe(s, LogisticRegression(class_weight='balanced', random_state=Config.SEED)), log_params)
SVMWithScalar = lambda s: SklearnModel("svm", text_and_scalar_pipe(s, LinearSVC(class_weight='balanced', random_state=Config.SEED)), svm_params)

MultiTextSVM = lambda texts: SklearnModel("svm", multi_text_pipe(texts, LinearSVC(class_weight='balanced', random_state=Config.SEED)), svm_params)
