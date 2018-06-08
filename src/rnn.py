import data_readers
import numpy as np
import os
import pickle
import random

from config import Config
from data_readers import read_dataset_splits
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional
from keras.models import Model
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

LABEL_TO_INDEX = {Config.LABEL_SHORT: 0, Config.LABEL_MEDIUM: 1, Config.LABEL_LONG: 2}
INDEX_TO_LABEL = {v:k for k,v in LABEL_TO_INDEX.items()}

def prepare_data(data):
    y = data.response_time_sec.apply(get_response_time_label).apply(lambda label: LABEL_TO_INDEX[label]).values
    X = data.question.apply(lambda sent: " ".join(sent)).values #drop(columns="response_time_sec").to_dict(orient="list")
    return X, y

def randvec(w, n=50, lower=-1.0, upper=1.0):
    """Returns a random vector of length `n`. `w` is ignored."""
    return np.array([random.uniform(lower, upper) for i in range(n)])

def fastTextVecLookup():
    FASTTEXT = {}
    with Config.FASTTEXT_FILE.open() as f:
        for line in f:
            line = line.strip().split()
            try:
                word = line[0]
                vec = np.array(line[1: ], dtype=np.float)
                FASTTEXT[word] = vec
            except:
                pass
    def fastTextVec(w):
        return FASTTEXT.get(w, randvec(w, n=EMBEDDING_DIM))

    return fastTextVec

EMBEDDINGS_FILE = Path("fasttext_emb_%d.pkl" % MAX_NUM_WORDS)
def getFastTextEmbeddings(word_index):
    if EMBEDDINGS_FILE.exists():
        with EMBEDDINGS_FILE.open(mode='rb') as f:
            return pickle.load(f)

    print("Initializing FastText Embeddings")
    lookup = fastTextVecLookup()

    num_words = max(word_index.values()) + 1
    assert num_words <= MAX_NUM_WORDS + 1
    matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        matrix[i] = lookup(word)

    with EMBEDDINGS_FILE.open(mode='wb') as f:
        pickle.dump(matrix, f)
    return matrix

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def simpleRNN(embeddings, hidden_dim=100):
    inputs = Input(shape=(MAX_QUESTION_LEN, ), dtype='int32')

    input_dim, output_dim = embeddings.shape
    x = Embedding(input_dim, output_dim,
                  weights = [embeddings],
                  input_length=MAX_QUESTION_LEN,
                  trainable=True)(inputs)
    x = Bidirectional(LSTM(hidden_dim))(x)
    preds = Dense(len(Config.LABELS), activation='sigmoid')(x)

    model = Model(inputs, preds)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    print(model.summary())
    return model

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
    data = read_dataset_splits(reader=data_readers.read_question_only_data, splits=["train", "dev"])
    X_train, y_train = prepare_data(data.train)
    X_dev, y_dev = prepare_data(data.dev)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", split=' ', lower=True)
    tokenizer.fit_on_texts(X_train)

    embeddings = getFastTextEmbeddings(tokenizer.word_index)

    model = simpleRNN(embeddings, hidden_dim=200)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_dev = tokenizer.texts_to_sequences(X_dev)

    X_train = pad_sequences(X_train, maxlen=MAX_QUESTION_LEN)
    X_dev = pad_sequences(X_dev, maxlen=MAX_QUESTION_LEN)

    model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_dev, y_dev))
    model.save("model.h5")

    train_preds = np.argmax(model.predict(X_train, batch_size=BATCH_SIZE), axis=1)
    dev_preds = np.argmax(model.predict(X_dev, batch_size=BATCH_SIZE), axis=1)

    evaluate(y_train, train_preds, name="train")
    evaluate(y_dev, dev_preds, name="dev")



