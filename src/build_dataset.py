import argparse
import data_readers
import data_util
import os
import pandas as pd
import progressbar

from config import Config
from console import log_info
from enum import Enum
from pathlib import Path

class Dataset(Enum):
    QUESTION_ONLY = 1

def build_question_only(split="tiny"):
    data = data_readers.read_corpus(split)
    questions = []
    response_times_sec = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    progress = progressbar.ProgressBar(max_value=len(sessions)).start()
    for i, session in enumerate(sessions):
        for question, response in session.iter_question_and_response():
            questions.append(question.text)
            response_times_sec.append((response.created_at - question.created_at).seconds)
            session_ids.append(session.id)
        progress.update(i)

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "response_time_sec": response_times_sec})
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

    log_info("Building the %s dataset" % args.dataset.name.lower())

    for split in Config.SPLITS:
        print("\tBuilding %s" % split)
        dataset = builders[args.dataset](split)
        print("\tExtracted %s samples" % dataset.shape[0])

        dest = get_dest_name(split)
        print("\tWriting dataset to %s" % dest)
        dataset.to_csv(dest, index=False)
