import ast
import numpy as np
import pandas as pd

from config import Config

def read_corpus(split=None):
    dtypes = {'session_id': np.int32, 'created_at': object, 'sent_from': str, 'sent_to': str, 'content_type': str}
    converters = {"text":ast.literal_eval}
    if split is None:
        path = Config.CORPUS_FILE
    else:
        path = Config.CORPUS_SPLIT_FILE(split)

    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, parse_dates=["created_at"], converters=converters)
    print("Read Preprocessed Data with %d rows" % data.shape[0])
    return data

def read_question_response_time_sec_data():
    dtypes = {"response_time_sec": np.int32}
    converters = {"question": ast.literal_eval}
    data = pd.read_csv(Config.QUESTION_RESPONSE_TIME_SEC_DATASET_FILE, sep=",", header=0, dtype=dtypes, converters=converters)
    print("Read Question-Response Time Data with %d rows" % data.shape[0])
    return data

if __name__ == "__main__":
    #data = read_corpus()
    #data = read_question_response_time_sec_data()
    #import matplotlib.pyplot as plt
    #ta.hist(column="response_time_sec", bins=1000)
    #plt.show()

    data = read_corpus()
    print(len(data.session_id))
    print(len(data.session_id.unique()))

