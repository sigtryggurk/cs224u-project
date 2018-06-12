import data_readers
import numpy as np
import os
import random

from collections import Counter
from config import Config
from data_readers import read_dataset_splits, read_corpus
from model_utils import get_response_time_label, add_cosine_similarity, add_question_length, add_jensen_shannon
from scipy.stats import pearsonr

random.seed(Config.SEED)

def calc_correlation(data, col_name):
    return pearsonr(data.train['response_time_sec'], data.train[col_name])

if __name__ == '__main__':
    results = {}
    data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
    data = add_jensen_shannon(data)
    results['question_and_js'] = calc_correlation(data, 'jensen_shannon')
    
    df = read_corpus(split='train')
    all_words = [item for sublist in df.text for item in sublist]
    for N_words in [25, 50, 100]:
        top_words = [item[0] for item in Counter(all_words).most_common(N_words)]
        data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
        data = add_jensen_shannon(data, stopwords=top_words)
        results['question_and_js_top' + str(N_words)] = calc_correlation(data, 'jensen_shannon')
    
    data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
    data = add_cosine_similarity(data)
    results['question_and_similarity'] = calc_correlation(data, 'cosine_similarity')
    
    df = read_corpus(split='train')
    all_words = [item for sublist in df.text for item in sublist]
    for N_words in [25, 50, 100]:
        top_words = [item[0] for item in Counter(all_words).most_common(N_words)]
        data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
        data = add_cosine_similarity(data, stopwords=top_words)
        results['question_and_similarity_top' + str(N_words)] = calc_correlation(data, 'cosine_similarity')
    
    data = read_dataset_splits(reader=data_readers.read_question_only_data)
    data = add_question_length(data)
    results['question_and_length'] = calc_correlation(data, 'question_length')
    
    data = read_dataset_splits(reader=data_readers.read_question_and_sentiment_data)
    results['question_and_sentiment'] = calc_correlation(data, 'question_sentiment')
    
    data = read_dataset_splits(reader=data_readers.read_question_and_duration_data)
    results['question_and_duration'] = calc_correlation(data, 'question_duration_sec')
    
    data = read_dataset_splits(reader=data_readers.read_question_and_index_data)
    results['question_and_index'] = calc_correlation(data, 'question_index')
    
    with Config.RUNS_DIR.joinpath('correlations').open(mode='w') as corr_file:
        for key, value in results.items():
            print('{}, {}'.format(key, value[0]), file=corr_file)
    
    
    
    

    

    
    


