import argparse
import data_readers
import numpy as np
import os

from config import Config
from pathlib import Path

def get_dest(split='train'):
    path = Path(Config.CORPUS_FILE)
    return os.path.join(Config.DATA_DIR, "{split}_{stem}{ext}".format(split=split, stem=path.stem, ext=path.suffix))

def split_data(data, tiny_f=0.01, train_f=0.7, dev_f=0.15, test_f=0.15):
    sessions = data.session_id.unique()
    assert abs(train_f + dev_f + test_f - 1.0) * len(sessions) < 1
    assert tiny_f < train_f

    print("Splitting %d sessions" % len(sessions))
    np.random.seed(seed=42)
    np.random.shuffle(sessions)
    train_i, dev_i, test_i = np.split(sessions, [int(train_f*len(sessions)), int((train_f+dev_f)*len(sessions))])
    tiny_i = train_i[:int(tiny_f * len(sessions))]
    print("\tTiny: %d\n\tTrain: %d\n\tDev: %d\n\tTest: %d" % (len(tiny_i),len(train_i), len(dev_i), len(test_i)))
    return data.iloc[tiny_i], data.iloc[train_i], data.iloc[dev_i], data.iloc[test_i]

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

    data = data_readers.read_corpus()

    tiny, train, dev, test = split_data(data, tiny_f=args.tiny, train_f=args.train, dev_f=args.dev, test_f=args.test)
    splits = {"tiny": tiny, "train": train, "dev": dev, "test": test}

    for name, split in splits.items():
        dest = get_dest(split=name)
        print("Writing %d %s sessions to %s" % (split.shape[0], name, dest))
        split.to_csv(dest, index=False)

