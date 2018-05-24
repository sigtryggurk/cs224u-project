import argparse
import data_readers
import data_util
import numpy as np
import os

from collections import defaultdict
from config import Config
from console import log_info
from pathlib import Path

def get_dest(split='train'):
    path = Path(Config.CORPUS_FILE)
    return os.path.join(Config.DATA_DIR, "{split}_{stem}{ext}".format(split=split, stem=path.stem, ext=path.suffix))

def get_num_questions_to_session_ids(questions_and_response_times):
    session_id_to_num_questions = defaultdict(int)
    for _, row in questions_and_response_times.iterrows():
        session_id_to_num_questions[row.session_id] += 1

    num_questions_to_session_ids = defaultdict(list)
    for session_id, num_questions in session_id_to_num_questions.items():
        num_questions_to_session_ids[num_questions].append(session_id)
    return num_questions_to_session_ids

def get_stratified_session_ids(num_questions_to_session_ids, min_f):
    groups = []
    current_group = []
    for session_ids in num_questions_to_session_ids.values():
        current_group.extend(session_ids)
        if len(current_group) * min_f > 1.0:
            groups.append(current_group)
            current_group = []
    groups[-1].extend(current_group)
    return groups

def split_data(data, tiny_f=0.01, train_f=0.7, dev_f=0.15, test_f=0.15):
    session_ids = data.session_id.unique()
    assert abs(train_f + dev_f + test_f - 1.0) * len(session_ids) < 1
    assert tiny_f < train_f
    log_info("Splitting %d session_ids" % len(session_ids))

    log_info("Extracting Questions")
    questions_and_response_times = data_util.get_questions_and_response_times(data)
    log_info("Extracted %d Questions from %d sessions" % (questions_and_response_times.shape[0], len(questions_and_response_times.session_id.unique())))

    session_id_to_num_questions = defaultdict(int)
    for _, row in questions_and_response_times.iterrows():
        session_id_to_num_questions[row.session_id] += 1

    num_questions_to_session_ids = defaultdict(list)
    for session_id, num_questions in session_id_to_num_questions.items():
        num_questions_to_session_ids[num_questions].append(session_id)

    groups = get_stratified_session_ids(num_questions_to_session_ids,  min([train_f, dev_f, test_f]))

    session_id_splits = defaultdict(list)
    for group in groups:
        np.random.seed(seed=42)
        np.random.shuffle(group)

        train_split, dev_split, test_split = np.split(group, [int(np.round(train_f * len(group))), int(np.round((train_f + dev_f) * len(group)))])
        session_id_splits["train"].extend(train_split)
        session_id_splits["dev"].extend(dev_split)
        session_id_splits["test"].extend(test_split)
    # tiny is a subset of train
    session_id_splits["tiny"] = session_id_splits["train"][:int(np.round(tiny_f * len(session_ids)))]
    for split, ids in session_id_splits.items():
        num_q = sum([session_id_to_num_questions[i] for i in ids])
        print("\t%s: %d sessions, %d questions" % (split, len(ids), num_q))

    return  data[data.session_id.isin(session_id_splits["tiny"])],\
            data[data.session_id.isin(session_id_splits["train"])],\
            data[data.session_id.isin(session_id_splits["dev"])],\
            data[data.session_id.isin(session_id_splits["test"])]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train" , dest="train", type=float,
            default = 0.7, help="Fraction of train data")
    parser.add_argument("-d", "--dev" , dest="dev", type=float,
            default = 0.15, help="Fraction of dev data")
    parser.add_argument("-e", "--test" , dest="test", type=float,
            default = 0.15, help="Fraction of test data")
    parser.add_argument("-y", "--tiny" , dest="tiny", type=float,
            default = 0.01, help="Fraction of tiny data")
    args = parser.parse_args()
    assert args.tiny < args.train

    log_info("Reading Corpus")
    data = data_readers.read_corpus()

    tiny, train, dev, test = split_data(data, tiny_f=args.tiny, train_f=args.train, dev_f=args.dev, test_f=args.test)
    splits = {"tiny": tiny, "train": train, "dev": dev, "test": test}

    for name, split in splits.items():
        dest = get_dest(split=name)
        log_info("Writing %d %s rows to %s" % (split.shape[0], name, dest))
        split.to_csv(dest, index=False)

