import argparse
import data_readers
import pandas as pd

from config import Config
from pathlib import Path
from sklearn.externals import joblib
from trainer import prepare_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clf_path", type=str, help="Path to the saved classifier")
    args = parser.parse_args()

    clf_path = Path(args.clf_path).resolve()
    clf = joblib.load(clf_path)

    question_and_duration = data_readers.read_question_and_duration_data(split="dev")
    question_and_response = data_readers.read_question_and_response_data(split="dev")
    assert all(question_and_duration.question == question_and_response.question)
    question_and_duration["response"] = question_and_response.response

    X, y = prepare_data(question_and_duration)
    preds = clf.predict(X)
    confidences = clf.decision_function(X)
    clf_classes = clf.classes_.tolist()
    short_i = clf_classes.index(Config.LABEL_SHORT)
    medium_i = clf_classes.index(Config.LABEL_MEDIUM)
    long_i = clf_classes.index(Config.LABEL_LONG)
    short_confs = confidences[:, short_i]
    medium_confs = confidences[:, medium_i]
    long_confs = confidences[:, long_i]

    res = pd.DataFrame.from_dict({"session_id": question_and_duration.session_id,
        "question":question_and_duration.question, "response": question_and_response.response, "question_duration_sec": question_and_duration.question_duration_sec, "response_time_sec": question_and_duration.response_time_sec, "predicted": preds, "true": y,
        "short_confidence": short_confs, "medium_confidence": medium_confs, "long_confidence": long_confs
        })

    res.to_csv("predicted.csv", index=False, header=True, columns=["session_id", "question", "question_duration_sec", "response", "response_time_sec", "predicted", "true", "short_confidence", "medium_confidence", "long_confidence"])
