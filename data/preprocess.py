import argparse
import os
import pandas as pd
import progressbar
import re
import spacy

from pathlib import Path

URL_TAG = "<url>"
REMOVED_ROWS_FILE = "removed_rows.csv"

def read_csv(datafile):
    data = pd.read_csv(datafile, sep=',', header=0)
    print("\tRead %d rows" % data.shape[0])
    return data

def utterance_equals(utterance1, utterance2):
    return utterance1.sent_from == utterance2.sent_from and \
            utterance1.sent_to == utterance2.sent_to and \
            utterance1.text == utterance2.text

def remove_rows(data, rows):
    header = True
    mode = 'w'
    if Path(REMOVED_ROWS_FILE).exists():
        header = False
        mode = 'a'
    print("\tOutputting %d removed rows to %s" % (len(rows), REMOVED_ROWS_FILE))
    data.iloc[rows].to_csv(REMOVED_ROWS_FILE, header=header, mode=mode)
    data.drop(index=rows, inplace=True)
    data.reset_index(inplace=True, drop=True)

def dedupe_utterances(data):
    nrows, _ = data.shape
    i = 0
    rows_to_delete = []
    progress = progressbar.ProgressBar(max_value=nrows).start()
    while i < nrows:
        progress.update(i)
        delta = 1
        row = data.iloc[i]
        while i + delta < nrows and utterance_equals(row, data.iloc[i+delta]):
            delta += 1

        # retain only first instance of utterance
        rows_to_delete.extend(list(range(i+1, i+delta)))

        i += delta
    progress.finish()

    remove_rows(data, rows_to_delete)

    return data

def remove_invalid_rows(data):
    """
      Removes invalid rows from data.

      A row is invalid if
          * the text is NaN
    """
    progress = progressbar.ProgressBar(max_value=data.shape[0]).start()
    invalid_utterances = []
    for i, row in data.iterrows():
        if type(row.text) != str:
            invalid_utterances.append(i)
        progress.update(i)
    progress.finish()

    remove_rows(data, invalid_utterances)
    return data

def normalize_url(data):
    progress = progressbar.ProgressBar(max_value=data.shape[0]).start()
    total_num_subs = 0
    for i, row in data.iterrows():
        normalized, num_subs = re.subn('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', URL_TAG, row.text)
        if num_subs > 0:
            data.at[i,'text'] = normalized

        progress.update(i)
        total_num_subs += num_subs

    progress.finish()
    print("\tNormalized %d URLs" % total_num_subs)
    return data

def parse_timestamps(data):
    data.created_at = pd.to_datetime(data.created_at,format="%Y-%m-%d %H:%M:%S %Z")
    return data

def tokenize_utterances(data):
    tokenizer = spacy.load('en_core_web_sm', disable=["tagger", "parser", "ner", "textcat"])
    tokenizer.tokenizer.add_special_case(URL_TAG, [{spacy.symbols.ORTH: URL_TAG}])

    progress = progressbar.ProgressBar(max_value=data.shape[0]).start()
    def tokenize(text):
        progress.update(progress.value+1)
        return [token.string for token in tokenizer(text)]

    data.text = data.text.apply(tokenize)
    progress.finish()
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str, help="Path to the datafile to process")
    parser.add_argument("-d", "--dest", dest="dest", type=str, default=None,
            help="Path to destination file. Defaults to {datafile}_processed.csv")
    args = parser.parse_args()

    path = Path(args.datafile)
    assert path.exists() and path.is_file() and path.suffix == '.csv'

    if args.dest is None:
        args.dest = path.stem + "_preprocessed" + path.suffix

    if Path(REMOVED_ROWS_FILE).exists():
        print("Deleting %s" % REMOVED_ROWS_FILE)
        os.remove(REMOVED_ROWS_FILE)

    print("Reading CSV file")
    data = read_csv(args.datafile)

    print("Deduping utterances")
    data = dedupe_utterances(data)

    print("Removing invalid rows")
    data = remove_invalid_rows(data)

    print("Normalizing urls")
    data = normalize_url(data)

    print("Parsing timestamps")
    data = parse_timestamps(data)

    print("Tokenizing utterances")
    date = tokenize_utterances(data)

    print("Writing to %s" % args.dest)
    data.to_csv(args.dest, index=False)
    print("\tWrote %d rows" % data.shape[0])
