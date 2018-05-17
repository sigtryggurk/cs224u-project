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



if __name__ == "__main__":
    data = read_preprocessed_data()
