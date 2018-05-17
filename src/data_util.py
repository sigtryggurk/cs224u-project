import ast
import numpy as np
import pandas as pd

from config import Config

def read_preprocessed_data():
    dtypes = {'session_id': np.int32, 'created_at': object, 'sent_from': str, 'sent_to': str, 'content_type': str}
    converters = {"text":ast.literal_eval}
    data = pd.read_csv(Config.preprocessed_data_file, sep=",", header=0, dtype=dtypes, parse_dates=["created_at"],
            converters=converters)
    print("Read Preprocessed Data with %d rows" % data.shape[0])
    return data

def read_question_response_time_sec_data():
    dtypes = {"response_time_sec": np.int32}
    converters = {"question": ast.literal_eval}
    data = pd.read_csv(Config.question_response_time_sec_dataset_file, sep=",", header=0, dtype=dtypes, converters=converters)
    print("Read Question-Response Time Data with %d rows" % data.shape[0])
    return data

if __name__ == "__main__":
    #data = read_preprocessed_data()
    data = read_question_response_time_sec_data()
    print(data.iloc[0], data.question[0][3], data.response_time_sec[0])
