import data_readers
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

from collections import defaultdict
from config import Config
from data_readers import read_dataset_splits
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from model_utils import get_response_time_label
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score

MAX_NUM_WORDS = 40000
MAX_QUESTION_LEN = 128
EMBEDDING_DIM = 300
NUM_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DECAY = LEARNING_RATE / NUM_EPOCHS

LABEL_TO_INDEX = {Config.LABEL_SHORT: 0, Config.LABEL_LONG: 1}
def prepare_data(data):
    y = data.response_time_sec.apply(get_response_time_label).apply(lambda label: LABEL_TO_INDEX[label]).values
    X = data.question.apply(lambda sent: " ".join(sent)).values #drop(columns="response_time_sec").to_dict(orient="list")
    return X, y

def evaluate(y_true, y_pred, name="tiny"):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    directory = Path(os.path.join(Config.RUNS_DIR, "question_only", "simple_rnn", name))
    directory.mkdir(parents=True, exist_ok=True)

    with directory.joinpath("results").open(mode='w') as results_file:
        print("Precision: %.4f" % (precision if precision is not None else -1), file=results_file)
        print("Recall: %.4f" % (recall if recall is not None else -1), file=results_file)
        print("F1-score: %.4f" % (f1 if f1 is not None else -1), file=results_file)
        print("Support: %d" % (support if support is not None else -1), file=results_file)

    report = classification_report(y_true, y_pred)
    with directory.joinpath("classification_report").open(mode='w') as report_file:
        print(report, file=report_file)

    cm = confusion_matrix(y_true, y_pred)
    with directory.joinpath("confusion_matrix").open(mode='w') as cm_file:
        np.savetxt(cm_file, cm, fmt="%d")


if __name__ == "__main__":
    data = read_dataset_splits(reader=data_readers.read_question_only_data, splits=["tiny", "train", "test"])
    X_train, y_train = prepare_data(data.train)
    X_test, y_test = prepare_data(data.test)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", split=' ', lower=True)
    tokenizer.fit_on_texts(X_train)

    model = load_model("model.05-0.67.h5")

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=MAX_QUESTION_LEN)
 
    test_preds = np.rint(model.predict(X_test, batch_size=BATCH_SIZE))

    evaluate(y_test, test_preds, name="test")



