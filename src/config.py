from os.path import dirname, realpath, join

class Config:
    BASE_DIR = realpath(join(dirname(realpath(__file__)), '..'))
    DATA_DIR = join(BASE_DIR, "data")
    CORPUS_FILE = join(DATA_DIR, "yup_messages_preprocessed.csv")
    BASELINE_PREDS_FILE = join(DATA_DIR, "dev_baseline_predictions_logreg.csv")
    SPLITS = ["tiny", "train", "dev", "test"]
    def _corpus_split_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_yup_messages_preprocessed.csv" % split)
    CORPUS_SPLIT_FILE = _corpus_split_file

    def _question_only_dataset_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_question_only_dataset.csv" % split)
    QUESTION_ONLY_DATASET_FILE = _question_only_dataset_file

    def _question_and_index_dataset_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_question_and_index_dataset.csv" % split)
    QUESTION_AND_INDEX_DATASET_FILE = _question_and_index_dataset_file

    def _question_and_duration_dataset_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_question_and_duration_dataset.csv" % split)
    QUESTION_AND_DURATION_DATASET_FILE = _question_and_duration_dataset_file

    def _question_text_and_response_text_dataset_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_question_text_and_response_text_dataset.csv" % split)
    QUESTION_TEXT_AND_RESPONSE_TEXT_DATASET_FILE = _question_text_and_response_text_dataset_file

    MAX_CONTEXT_WINDOW_SIZE = 10
    def _question_and_window_size_dataset_file(split):
        assert split in Config.SPLITS
        return join(Config.DATA_DIR, "%s_question_and_context_window_dataset.csv" % split)
    QUESTION_AND_CONTEXT_WINDOW_DATASET_FILE = _question_and_window_size_dataset_file

    EMPTY_TAG = "<empty>"
    URL_TAG = "<url>"
    REMOVED_ROWS_FILE = join(DATA_DIR, "removed_rows.csv")

    STUDENT_SPEAKERS = ["student"]
    TUTOR_SPEAKERS = ["tutor"]
    SYSTEM_SPEAKERS =  ["system info", "system alert", "system warn", "bot"]
    PLATFORM_SPEAKERS = TUTOR_SPEAKERS  + SYSTEM_SPEAKERS

    LABEL_SHORT = "short"
    LABEL_MEDIUM = "medium"
    LABEL_LONG = "long"
    LABELS = [LABEL_SHORT, LABEL_MEDIUM, LABEL_LONG]
    THRESHOLD_SHORT = 15
    THRESHOLD_MEDIUM = 45

    SEED = 42 #For reproducibility




