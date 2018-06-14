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

    window_size = 5
    question_and_context = data_readers.read_question_and_context_data(split="dev", window_size=window_size, include_question_text=True, include_context_text=False, include_context_times=True)
    question_and_response = data_readers.read_question_and_response_data(split="dev")
    assert all(question_and_context.question == question_and_response.question)
    question_and_context["response"] = question_and_response.response

    X, y = prepare_data(question_and_context)
    preds = clf.predict(X)
    confidences = clf.decision_function(X)
    clf_classes = clf.classes_.tolist()
    short_i = clf_classes.index(Config.LABEL_SHORT)
    medium_i = clf_classes.index(Config.LABEL_MEDIUM)
    long_i = clf_classes.index(Config.LABEL_LONG)
    short_confs = confidences[:, short_i]
    medium_confs = confidences[:, medium_i]
    long_confs = confidences[:, long_i]

    res_dict = {"session_id": question_and_context.session_id,
        "question":question_and_context.question, "response": question_and_response.response, "response_time_sec": question_and_context.response_time_sec, "predicted": preds, "true": y,
        "short_confidence": short_confs, "medium_confidence": medium_confs, "long_confidence": long_confs}
    turns = []
    for i in range(1, window_size + 1):
        turn = "turn_time-%d" % i
        turns.append(turn)
        res_dict[turn] = question_and_context[turn]

    res = pd.DataFrame.from_dict(res_dict)
    columns=["session_id", "question"]
    columns.extend(turns)
    columns.extend(["response", "response_time_sec", "predicted", "true", "short_confidence", "medium_confidence", "long_confidence"])
    res.to_csv("dev_question_and_context_time_%d_predictions.csv" % window_size, index=False, header=True, columns=columns)