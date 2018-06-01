# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:17:21 2018

@author: Jayadev Bhaskaran
"""

#Used only for data generation.
from stanfordcorenlp import StanfordCoreNLP
from data_readers import read_dataset_splits, read_question_only_data

import pandas as pd

#Replace with path to your Stanford CoreNLP folder.
nlp = StanfordCoreNLP(r'C:\Users\Jayadev Bhaskaran\Downloads\stanford-corenlp-full-2018-02-27')

def get_sentiment(row):
    sentence = " ".join(row['question']) #Won't reconstruct exactly, but a good approximation.
    try:
        retval = int(nlp._request('tokenize,ssplit,parse,sentiment', sentence)['sentences'][0]['sentimentValue'])
    except:
        #There are about 40 sentences overall that CoreNLP barfs on (encoding issues).
        #Treat these as neutral.
        print(sentence)
        retval = 2
        
    return retval

data = read_dataset_splits(reader=read_question_only_data)
tiny = data['tiny']
train = data['train']
dev = data['dev']
test = data['test']

tiny['sentiment'] = tiny.apply(lambda row: get_sentiment(row), axis=1)
train['sentiment'] = train.apply(lambda row:get_sentiment(row), axis=1)
dev['sentiment'] = dev.apply(lambda row:get_sentiment(row), axis=1) 
test['sentiment'] = test.apply(lambda row:get_sentiment(row), axis=1)

tiny.to_csv(Config.QUESTION_AND_SENTIMENT_DATASET_FILE('tiny'))
train.to_csv(Config.QUESTION_AND_SENTIMENT_DATASET_FILE('train'))
dev.to_csv(Config.QUESTION_AND_SENTIMENT_DATASET_FILE('dev'))
test.to_csv(Config.QUESTION_AND_SENTIMENT_DATASET_FILE('test'))

nlp.close()