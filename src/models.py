import copy
import numpy as np
import os
import random

from config import Config
from data_readers import read_question_only_data, read_dataset_splits
from datetime import datetime
from model_utils import extend_question_class, add_classes, plot_cm, dummy_tokenizer
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

random.seed(Config.SEED)

class SklearnTrainer(object):
    def __init__(self, model, params_range, n_samples, model_name, data_name):
        self.model = model
        self.params_sampler = ParameterSampler(params_range, n_iter=n_samples, random_state=Config.SEED)
        self.directory = os.path.join(Config.RUNS_DIR, self.data_name, self.model_name, str(datetime.now()))
        self.directory = Path(self.directory).mkdir(parents=True, exist_ok=True)

    def train(self, train_data, dev_data):
        X_train, y_train = train_data
        X_dev, y_dev = dev_data

        self.best_clf = None
        self.best_f1 = 0
        self.best_params = None

        for params in self.params_sampler:
            clf = self.model(**params)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_dev)
            f1 = f1_score(y_dev, preds, average='weighted')
            if f1 > self.best_f1:
                self.best_clf = clf
                self.best_f1 = f1
                self.best_params = params

        self.eval(X_train, y_train, split="train")
        self.eval(X_dev, y_dev, split="dev")

    def eval(self, X, y, split="tiny"):
        preds = self.best_clf.predict(X)

        split_dir = self.directory.joinpath(split)

        precision, recall, f1, support = precision_recall_fscore_support(y, preds, average='weighted')
        with split_dir.joinpath("results").open(mode='w') as results_file:
            print("Precision: %.4f" % precision, file=results_file)
            print("Recall: %.4f" % recall, file=results_file)
            print("F1-score: %.4f" % f1, file=results_file)
            print("Support: %.4f" % support, file=results_file)

        report = classification_report(y, preds)
        with split_dir.joinpath("classification_report").open(mode='w') as report_file:
            print(report, file=report_file)

        cm = confusion_matrix(y, preds)
        with split_dir.joinpath("confusion_matrix").open(mode='w') as cm_file:
            print(cm, file=cm_file)

        with split_dir.joinpath("params").open(mode='w') as params_file:
            print(self.best_params, file=params_file)


if __name__ == '__main__':
    params = {'clf__C': np.logspace(-4, 4, 10), 'clf__penalty': ['l2', 'l1']}

    model = lambda: Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(class_weight='balanced', random_state=Config.SEED))
    ])
    data = read_dataset_splits(reader=read_question_only_data)
    data = add_classes(data)

    train, dev, test = data['train'], data['dev'], data['test']

    trainer = SklearnTrainer(model, params, 5, "logistic_regression", "question_only")
