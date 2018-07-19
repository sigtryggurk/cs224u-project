import data_readers
import os
import pandas as pd
import random

from collections import Counter
from config import Config
from data_readers import read_dataset_splits, read_corpus
from functools import reduce
from model_utils import add_question_length, add_jensen_shannon
from pathlib import Path
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from trainer import prepare_data

random.seed(Config.SEED)

if __name__ == "__main__":    
    
    #q = data_readers.read_question_and_context_data(split = 'test', window_size=5, include_question_text=True, include_context_text=False, include_context_speaker=False, include_context_times=True)
    #data_name = 'question_and_context_time_5'
    #model_name = 'logistic'
    
    #q_length = read_dataset_splits(reader=data_readers.read_question_only_data, splits=['test'])
    #q_length = add_question_length(q_length)
    #q_duration = read_dataset_splits(reader=data_readers.read_question_and_duration_data, splits=['test'])
    #q_context = read_dataset_splits(reader=data_readers.read_question_and_context_data, splits=['test'], window_size=5, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=True)
    #arr = [q_length['test'], q_duration['test'], q_context['test']]
    #q = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), arr)
    #data_name = 'combined_length_ctime'
    #model_name = 'svm'
    
    q = data_readers.read_question_and_response_data(split="test")
    data_name = "question_and_response"
    model_name = "svm"
    
    clf_dir = Path(os.path.join(Config.RUNS_DIR, data_name, model_name))
    clf_path = clf_dir.joinpath("clf.pkl")
    clf = joblib.load(clf_path)
    
    X, y = prepare_data(q)
    preds = clf.predict(X)
    report = classification_report(y, preds, digits=4)
    with clf_dir.joinpath("classification_report_test_" + model_name).open(mode='w') as report_file:
        print(report, file=report_file)
