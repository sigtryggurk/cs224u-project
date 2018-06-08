# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:45:01 2018

@author: Jayadev Bhaskaran
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import base64
import itertools
import re

from unicodedata import normalize
from config import Config
from scipy import stats
from scipy.spatial.distance import cosine
from spacy.lang.en.stop_words import STOP_WORDS

SEED = Config.SEED
random.seed(SEED)

def get_response_time_label(time):
    if time < Config.THRESHOLD_SHORT:
        return Config.LABEL_SHORT
    elif time < Config.THRESHOLD_MEDIUM:
        return Config.LABEL_MEDIUM
    else:
        return Config.LABEL_LONG

def add_classes(data):
    '''
        Add label corresponding to question class.
    '''
    for key, value in data.items():
        value['question_class'] = value.apply(lambda row:
            get_response_time_label(row['response_time_sec']), axis=1)

    return data
            
def add_question_length(data):
    '''
        Add question length as a feature.
    '''
    for key, value in data.items():
        value['question_length'] = value.apply(lambda row:
            len(row['question']), axis=1)
    
    return data

def calc_cosine_similarity(row, stopwords=STOP_WORDS):
    odd_text = []
    even_text = []
    for i in range(1,10,2):
        odd_text += row['turn_text-' + str(i)]
        even_text += row['turn_text-' + str(i+1)]
        
    odd_total = [item for sublist in odd_text for item in sublist]
    even_total = [item for sublist in even_text for item in sublist]
    
    odd_vector = [odd_total.count(w) for w in stopwords] 
    even_vector = [even_total.count(w) for w in stopwords]
    
    if np.linalg.norm(odd_vector) == 0 or np.linalg.norm(even_vector) == 0:
        return 0
    else:
        return 1 - cosine(odd_vector, even_vector) #Cosine similarity as cos(theta)  
    
def add_cosine_similarity(data, stopwords=STOP_WORDS):
    '''
        Add cosine distance for stop word vectors as a feature.
    '''
    for key, value in data.items():
        value['cosine_similarity'] = value.apply(lambda row:
            calc_cosine_similarity(row, stopwords=stopwords), axis=1)
    
    return data

def plot_cm(cm, filename):
    '''
        Takes in a confusion matrix and saves it as a PNG image.
    '''
    plt.imshow(cm, cmap=plt.cm.Reds)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=6)
    plt.xlabel('Predicted label', fontsize=6)
    plt.xticks(np.arange(len(Config.LABELS)), Config.LABELS, fontsize=8)
    plt.yticks(np.arange(len(Config.LABELS)), Config.LABELS, fontsize=8)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.savefig(filename, dpi=300)
    plt.close()

def dummy_tokenizer(tokens):
    return tokens

def get_question_length(X):
    mylen = np.vectorize(len)
    return mylen(X).reshape(-1, 1)
