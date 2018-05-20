from os.path import dirname, realpath, join

class Config:
    BASE_DIR = realpath(join(dirname(realpath(__file__)), '..'))
    DATA_DIR = join(BASE_DIR, "data")
    PREPROCESSED_DATA_FILE = join(DATA_DIR, "yup_messages_preprocessed.csv")
    QUESTION_RESPONSE_TIME_SEC_DATASET_FILE = join(DATA_DIR, "question_response_time_sec_dataset.csv")

    URL_TAG = "<url>"
    REMOVED_ROWS_FILE = join(DATA_DIR, "removed_rows.csv")

