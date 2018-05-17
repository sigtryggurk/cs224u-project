import argparse
import ast
import numpy as np
import os
import pandas as pd
import progressbar
import sys

from enum import Enum
from pathlib import Path

PREPROCESSED_DATA_FILE = "yup_messages_preprocessed.csv"

class Dataset(Enum):
    QUESTION_RESPONSE_TIME_SEC = 1

def read_preprocessed_data():
    dtypes = {'session_id': np.int32, 'created_at': object, 'sent_from': str, 'sent_to': str, 'content_type': str}
    converters = {"text": ast.literal_eval}
    data = pd.read_csv(PREPROCESSED_DATA_FILE, sep=",", header=0, dtype=dtypes, parse_dates=["created_at"], converters=converters)
    print("Read Preprocessed Data with %d rows" % data.shape[0])
    return data

def is_student_text(row):
    return row.sent_from == "student"

def is_tutor_text(row):
    return row.sent_from == "tutor"

def is_tutor_question(row):
    return is_tutor_text(row) and '?' in row.text

def build_question_only():
    questions = []
    response_times_sec = []

    data = read_preprocessed_data()
    nrows = data.shape[0]
    progress = progressbar.ProgressBar(max_value=nrows).start()
    i = 0
    while i < nrows:
        row = data.iloc[i]

        if is_tutor_question(row) and i+1 < nrows:
            question = row.text
            if len(question) < 2 and i > 0 and is_tutor_text(data.iloc[i-1]):
                question = data.iloc[i-1].text + question

            question_time = row.created_at
            response_time = None
            session_id = row.session_id

            i += 1
            row = data.iloc[i]
            while row.session_id == session_id and response_time is None:
                if is_student_text(row):
                    response_time = row.created_at
                elif is_tutor_text(row):
                    # extend question and update question time if the tutor follows up
                    question += row.text
                    question_time = row.created_at

                if i + 1 >= nrows:
                    break
                i += 1
                row = data.iloc[i]

            if len(question) > 1 and question_time is not None and response_time is not None:
                questions.append(question)
                response_times_sec.append((response_time - question_time).seconds)
        else:
            i += 1

        progress.update(i)

    dataset = pd.DataFrame.from_dict({"question": questions, "response_time_sec": response_times_sec})
    progress.finish()
    return dataset

if __name__ == "__main__":
    assert Path(PREPROCESSED_DATA_FILE).exists(), "%s does not exist" % PREPROCESSED_DATA_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", type=Dataset, default=Dataset.QUESTION_ONLY,
            help="Which dataset to build. Defaults to QUESTION_ONLY")
    args = parser.parse_args()

    dest = "%s_dataset.csv" % args.dataset.name.lower()
    if Path(dest).exists():
        delete = input("%s already exists. Do you wish to overwrite it? (y/n): " % dest)
        while delete.lower() not in ['y', 'n']:
            delete = input("%s is not a valid answer. Please type either 'y' or 'n'" % delete)
        if delete =='y':
            os.remove(dest)
        elif delete == 'n':
            sys.exit(0)

    builders = {Dataset.QUESTION_ONLY: build_question_only}

    dataset = builders[args.dataset]()
    print("Extracted %s samples" % dataset.shape[0])

    print("Writing dataset to %s" % dest)
    dataset.to_csv(dest, index=False)
