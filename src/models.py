import numpy as np

from config import Config
from model_utils import dummy_tokenizer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

class SklearnModel(object):
    def __init__(self, name, pipe, params_range):
        self.name = name
        self.pipe = pipe
        self.params_range = params_range

pipe = Pipeline([
             ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
             ('tfidf', TfidfTransformer()),
             ('clf', LogisticRegression())
             ])
params = {'clf__C': np.logspace(-4, 4, 100), 'clf__penalty': ['l2', 'l1']}
LogisticRegression = SklearnModel("logistic_regression", pipe, params)

pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(class_weight='balanced', random_state=Config.SEED))
    ])
params = {'clf__C': np.logspace(-4,4,100),
          'clf__loss': ['hinge', 'squared_hinge']}
SVM = SklearnModel("svm", pipe, params)

pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),
            ('clf', DummyClassifier(random_state=Config.SEED))
            ])
params = {}
Dummy = SklearnModel("dummy", pipe, params)
