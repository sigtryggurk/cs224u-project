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
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.optimizers import Adam
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

class F1_Score(Callback):
    def __init__(self, training_data):
        self.training_data = training_data
        super().__init__() 

    def on_train_begin(self, logs={}):
        self.train_f1s = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        train_predict = np.rint(self.model.predict(self.training_data[0]))
        train_targ = self.training_data[1]
        _train_f1 = f1_score(train_targ, train_predict, average='weighted')
        self.train_f1s.append(_train_f1)
        print(" — train_f1: %.4f" % _train_f1)
        
        val_predict = np.rint(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        self.val_f1s.append(_val_f1)
        print(" — val_f1: %.4f" % _val_f1)
        return

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

def simpleRNN(embeddings, hidden_dim=100, dropout=0.1, recurrent_dropout=0.0):



    inputs = Input(shape=(MAX_QUESTION_LEN, ), dtype='int32')

    input_dim, output_dim = embeddings.shape
    x = Embedding(input_dim, output_dim,
                  weights = [embeddings],
                  input_length=MAX_QUESTION_LEN,
                  trainable=True)(inputs)
    x = LSTM(hidden_dim,activation="sigmoid", recurrent_activation="hard_sigmoid", return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = LSTM(hidden_dim,activation="sigmoid", recurrent_activation="hard_sigmoid")(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, preds)

    opt = Adam(lr=LEARNING_RATE, decay=DECAY)  
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

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
    data = read_dataset_splits(reader=data_readers.read_question_only_data, splits=["tiny", "train", "dev"])
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

#    class_weight = defaultdict(int)
#    for label in y_train:
#        class_weight[label] += 1
#    tot = sum(class_weight.values())
#    for c in class_weight.keys():
#        class_weight[c] /= tot

    callbacks = [F1_Score((X_train, y_train)),
                 ModelCheckpoint("model.{epoch:02d}-{val_loss:.2f}.h5", monitor="val_loss", save_best_only=False, period=1)]
    

    res = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_dev, y_dev), callbacks=callbacks)

    with open("history.pkl", "wb") as history:
        pickle.dump(res.history, history)

    # summarize history for loss
    plt.plot(res.history['loss'])
    plt.plot(res.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig('loss.png')
 
    train_preds = np.rint(model.predict(X_train, batch_size=BATCH_SIZE))
    dev_preds = np.rint(model.predict(X_dev, batch_size=BATCH_SIZE))

    evaluate(y_train, train_preds, name="train")
    evaluate(y_dev, dev_preds, name="dev")



