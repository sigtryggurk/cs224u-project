import ast
import numpy as np
import pandas as pd

from config import Config
from console import log_info

def read_corpus(split=None):
    dtypes = {'session_id': np.int32, 'created_at': object, 'sent_from': str, 'sent_to': str, 'content_type': str}
    converters = {"text":ast.literal_eval}
    if split is None:
        path = Config.CORPUS_FILE
        split = "entire"
    else:
        path = Config.CORPUS_SPLIT_FILE(split)

    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, parse_dates=["created_at"], converters=converters)

    log_info("Read %s corpus with %d rows" % (split, data.shape[0]))
    return data

def read_question_only_data(split="tiny"):
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32}
    converters = {"question": ast.literal_eval}
    data = pd.read_csv(Config.QUESTION_ONLY_DATASET_FILE(split), sep=",", header=0, dtype=dtypes, converters=converters)
    log_info("Read %s_question_only data with %d rows" % (split, data.shape[0]))
    return data

def read_dataset_splits(splits=Config.SPLITS, reader=read_question_only_data):
    data = {}
    for split in splits:
        data[split] = reader(split)
    return data


if __name__ == "__main__":
    data = read_dataset_splits()
    print(data.keys())
    #data = read_question_only_data(split="train")
    #import matplotlib.pyplot as plt
    #plt.figure()
    #data.response_time_sec.plot.hist(bins=1000)
    #plt.show()

