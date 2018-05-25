from os.path import dirname, realpath, join

class Config:
    BASE_DIR = realpath(join(dirname(realpath(__file__)), '..'))
    DATA_DIR = join(BASE_DIR, "data")
    CORPUS_FILE = join(DATA_DIR, "yup_messages_preprocessed.csv")
    SPLITS = ["tiny", "train", "dev", "test"]
    def _corpus_split_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_yup_messages_preprocessed.csv" % split)
    CORPUS_SPLIT_FILE = _corpus_split_file

    def _question_only_dataset_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_question_only_dataset.csv" % split)
    QUESTION_ONLY_DATASET_FILE = _question_only_dataset_file

    def _question_text_and_response_text_dataset_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_question_text_and_response_text_dataset.csv" % split)
    QUESTION_TEXT_AND_RESPONSE_TEXT_DATASET_FILE = _question_text_and_response_text_dataset_file

    EMPTY_TAG = "<empty>"
    URL_TAG = "<url>"
    REMOVED_ROWS_FILE = join(DATA_DIR, "removed_rows.csv")

    LABEL_SHORT = "short"
    LABEL_MEDIUM = "medium"
    LABEL_LONG = "long"
    LABELS = [LABEL_SHORT, LABEL_MEDIUM, LABEL_LONG]
    THRESHOLD_SHORT = 15
    THRESHOLD_MEDIUM = 45

    SEED = 42 #For reproducibility




