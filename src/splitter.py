import argparse
import data_readers
import numpy as np
import os

from config import Config
from pathlib import Path

def get_dest(split='train'):
    path = Path(Config.PREPROCESSED_DATA_FILE)
    return os.path.join(Config.DATA_DIR, "{split}_{stem}{ext}".format(split=split, stem=path.stem, ext=path.suffix))

def train_dev_test_split(data, train=0.7, dev=0.15, test=0.15):
    assert abs(train + dev + test - 1.0) * data.shape[0] < 1
    print("Splitting %d rows" % data.shape[0])
    train, dev, test = np.split(data.sample(frac=1, random_state=42), [int(train*len(data)), int((train+dev)*len(data))])
    print("\tTrain: %d\n\tDev: %d\n\tTest: %d" % (train.shape[0], dev.shape[0], test.shape[0]))
    return train, dev, test

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

    data = data_readers.read_preprocessed_data()
    nrows = data.shape[0]
    assert abs(args.train + args.dev + args.test - 1.0) * nrows < 1

    train, dev, test = train_dev_test_split(data, train=args.train, dev=args.dev, test=args.test)
    tiny = train.iloc[:int(args.tiny * nrows)]

    splits = {"tiny": tiny, "train": train, "dev": dev, "test": test}

    for name, split in splits.items():
        dest = get_dest(split=name)
        print("Writing %d %s sessions to %s" % (split.shape[0], name, dest))
        split.to_csv(dest, index=False)

