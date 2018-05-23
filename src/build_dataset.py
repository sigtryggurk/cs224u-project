import argparse
import data_readers
import os
import pandas as pd
import progressbar

from config import Config
from enum import Enum
from pathlib import Path

class Dataset(Enum):
    QUESTION_ONLY = 1

def is_student_text(row):
    return row.sent_from == "student"

def is_tutor_text(row):
    return row.sent_from == "tutor"

def is_tutor_question(row):
    return is_tutor_text(row) and '?' in row.text

def build_question_only(split="tiny"):
    questions = []
    response_times_sec = []

    data = data_readers.read_corpus(split)
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

def get_dest_name(split="tiny"):
    destname = "%s_%s_dataset.csv" % (split, args.dataset.name.lower())
    dest = os.path.join(Config.DATA_DIR, destname)
    return dest

if __name__ == "__main__":
    assert Path(Config.CORPUS_FILE).exists(), "%s does not exist" % Config.CORPUS_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", type=Dataset, default=Dataset.QUESTION_ONLY,
            help="Which dataset to build. Defaults to QUESTION_ONLY")
    args = parser.parse_args()

    builders = {Dataset.QUESTION_ONLY: build_question_only}

    print("Building the %s dataset" % args.dataset.name.lower())

    for split in Config.SPLITS:
        print("\tBuilding %s" % split)
        dataset = builders[args.dataset](split)
        print("\tExtracted %s samples" % dataset.shape[0])

        dest = get_dest_name(split)
        print("\tWriting dataset to %s" % dest)
        dataset.to_csv(dest, index=False)
