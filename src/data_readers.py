import ast
import numpy as np
import pandas as pd

from config import Config
from console import log_info
from dotdict import DotDict

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
    path = Config.QUESTION_ONLY_DATASET_FILE(split)
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)
    log_info("Read %s data with %d rows" % (path.stem, data.shape[0]))

    return data

def read_question_and_newlines_data(split="tiny"):
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32}
    converters = {"question": ast.literal_eval}
    path = Config.QUESTION_AND_NEWLINES_DATASET_FILE(split)
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)
    log_info("Read %s data with %d rows" % (path.stem, data.shape[0]))
    return data

def read_question_and_index_data(split="tiny"):
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32, "question_index": np.int32}
    converters = {"question": ast.literal_eval}
    path = Config.QUESTION_AND_INDEX_DATASET_FILE(split)
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)
    log_info("read %s data with %d rows" % (path.stem, data.shape[0]))
    return data

def read_question_and_duration_data(split="tiny"):
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32, "question_duration_sec": np.int32}
    converters = {"question": ast.literal_eval}
    path = Config.QUESTION_AND_DURATION_DATASET_FILE(split)
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)
    log_info("read %s data with %d rows" % (path.stem, data.shape[0]))
    return data

def read_question_and_response_data(split="tiny"):
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32}
    converters = {"question": ast.literal_eval, "response": ast.literal_eval}
    path = Config.QUESTION_TEXT_AND_RESPONSE_TEXT_DATASET_FILE(split)
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)
    log_info("Read %s data with %d rows" % (path.stem, data.shape[0]))
    return data

def read_question_and_sentiment_data(split="tiny"):
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32, "sentiment": np.int32}
    converters = {"question": ast.literal_eval}
    path = Config.QUESTION_AND_SENTIMENT_DATASET_FILE(split)
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)
    log_info("read %s data with %d rows" % (path.stem, data.shape[0]))
    return data

def read_question_and_context_data(split="tiny", window_size=1, include_question_text=True, include_context_text=True, include_context_speaker=True, include_context_times=False):
    assert window_size <= Config.MAX_CONTEXT_WINDOW_SIZE
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32}
    converters = {}

    if include_context_speaker:
        for i in range(1, window_size+1):
            dtypes["turn_speaker-%d" % i] = str

    if include_context_times:
        for i in range(1, window_size+1):
            dtypes["turn_time-%d" % i] = np.float32

    if include_question_text:
        converters["question"] = ast.literal_eval

    if include_context_text:
        for i in range(1, window_size+1):
            converters["turn_text-%d" % i] = ast.literal_eval

    path = Config.QUESTION_AND_CONTEXT_WINDOW_DATASET_FILE(split)
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)

    drop_columns = set(data.columns.values) - (set(dtypes.keys()) | set(converters.keys()))
    data.drop(labels=drop_columns, axis="columns", inplace=True)

    log_info("Read %s data with %d rows" % (path.stem, data.shape[0]))
    return data

def read_dataset_splits(splits=Config.SPLITS, reader=read_question_only_data, **kwargs):
    data = {}
    for split in splits:
        data[split] = reader(split, **kwargs)
    return DotDict(data)

if __name__ == "__main__":
    #data = read_corpus()
    #print(data.sent_from.unique())
    #data = read_question_and_sentiment_data(split="dev")
    #print(data.columns.values)
    #print(data.shape)
    #print(data.keys())
    #data = read_dataset_splits(reader=read_question_and_sentiment_data)
    X, y = read_question_and_context_data(split="tiny", label=True)
    #data = read_dataset_splits(reader = lambda split: read_question_and_context_data(split=split, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=True, include_context_times=True))
    #import matplotlib.pyplot as plt
    #plt.figure()
    #data.response_time_sec.plot.hist(bins=1000)
    #plt.show()

