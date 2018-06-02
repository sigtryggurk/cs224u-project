import numpy as np
import os
import random

from config import Config
from data_readers import read_question_only_data, read_dataset_splits
from model_utils import dummy_tokenizer
from pathlib import Path
from progressbar import progressbar
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

random.seed(Config.SEED)

class SklearnTrainer(object):
    def __init__(self, model, params_range, n_samples, model_name, data_name):
        self.model = model
        self.params_sampler = ParameterSampler(params_range, n_iter=n_samples, random_state=Config.SEED)
        self.directory = Path(os.path.join(Config.RUNS_DIR, data_name, model_name))
        self.directory.mkdir(parents=True, exist_ok=True)

    def train(self, train_data, dev_data):
        X_train, y_train = train_data
        X_dev, y_dev = dev_data
        self.best_clf = None
        self.best_params = None
        best_f1 = 0

        for params in progressbar(self.params_sampler):
            clf = self.model.set_params(**params)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_dev)
            f1 = f1_score(y_dev, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                self.best_clf = clf
                self.best_params = params

        self.eval(X_train, y_train, split="train")
        self.eval(X_dev, y_dev, split="dev")

    def eval(self, X, y, split="tiny"):
        preds = self.best_clf.predict(X)
        split_dir = self.directory.joinpath(split)
        split_dir.mkdir(parents=True, exist_ok=True)

        precision, recall, f1, support = precision_recall_fscore_support(y, preds, average='weighted')
        with split_dir.joinpath("results").open(mode='w') as results_file:
            print("Precision: %.4f" % (precision if precision is not None else -1), file=results_file)
            print("Recall: %.4f" % (recall if recall is not None else -1), file=results_file)
            print("F1-score: %.4f" % (f1 if f1 is not None else -1), file=results_file)
            print("Support: %d" % (support if support is not None else -1), file=results_file)

        report = classification_report(y, preds)
        with split_dir.joinpath("classification_report").open(mode='w') as report_file:
            print(report, file=report_file)

        cm = confusion_matrix(y, preds)
        with split_dir.joinpath("confusion_matrix").open(mode='w') as cm_file:
            print(cm, file=cm_file)

        with split_dir.joinpath("params").open(mode='w') as params_file:
            print(self.best_params, file=params_file)

if __name__ == '__main__':
    params = {'clf__C': np.logspace(-4, 4, 100), 'clf__penalty': ['l2', 'l1']}

    model = Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(class_weight='balanced', random_state=Config.SEED))
    ])
    data = read_dataset_splits(reader=read_question_only_data, prepare=True)
    trainer = SklearnTrainer(model, params, 5, "logistic_regression", "question_only")

    trainer.train(data.train, data.dev)
