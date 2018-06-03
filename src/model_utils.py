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

_punctuation_re = re.compile(r'[\t !"#$%&\'()*\-/<=>?@\[\\\]^_`{|},.]+')

def slugify(text, delim='-'):
    """
    Generate an ASCII-only slug.
    """
    result = []
    for word in _punctuation_re.split(text.lower()):
        word = normalize('NFKD', word) \
               .encode('ascii', 'ignore') \
               .decode('utf-8')

        if word:
            result.append(word)

    return delim.join(result)

def plot_cm(cm, title="Confusion Matrix"):
    '''
        Takes in a confusion matrix and saves it as a PNG image.
    '''
    plt.imshow(cm, cmap=plt.cm.Reds)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=6)
    plt.xlabel('Predicted label', fontsize=6)
    plt.xticks(np.arange(len(Config.LABELS)), Config.LABELS, fontsize=8)
    plt.yticks(np.arange(len(Config.LABELS)), Config.LABELS, fontsize=8)
    plt.title(title, fontsize=8)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.savefig("cm_{}.png".format(slugify(title)), dpi=300)
    plt.close()

def dummy_tokenizer(tokens):
    return tokens

def get_question_length(X):
    mylen = np.vectorize(len)
    return mylen(X).reshape(-1, 1)
