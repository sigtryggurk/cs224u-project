# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:39:00 2018

@author: Jayadev Bhaskaran
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer

from data_readers import read_question_only_data, read_dataset_splits
from config import Config
from model_utils import extend_question_class, add_classes, plot_cm, dummy_tokenizer, get_question_length

import random

SEED = Config.SEED
random.seed(SEED)
    
def run_with_question_length(data):
    '''
        Input: Dictionary of data (tiny, train, dev, test)
        
        Does the usual countvectorizer/tfidf, and also includes 
        question length as another feature.
    '''
    train, dev, test = data['train'], data['dev'], data['test']    
    models = {}    

    #Logistic regression
    params = dict([
        ('clf__C', [0.01, 0.1, 1, 10, 100]),
        ('clf__penalty', ['l2', 'l1']),        
    ])

    pipe = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
                    ('tfidf', TfidfTransformer()),
                ])),
                ('length', FunctionTransformer(get_question_length, validate=False)),
            ])),
            ('clf', LogisticRegression(class_weight='balanced', random_state=SEED))            
    ])
    
    best_f1 = 0
    best_grid = {}
    report = []
    cm = []
    for g in ParameterGrid(params): 
        pipe.set_params(**g)
        pipe.fit(train['question'], train['question_class'])
        preds = pipe.predict(dev['question'])
        p, r, f, s = precision_recall_fscore_support(dev['question_class'], preds, average='weighted')  
        print(g)
        print(f)
        if f > best_f1:
            best_f1 = f
            best_grid = g
            report = classification_report(dev['question_class'], preds)
            cm = confusion_matrix(dev['question_class'], preds)
    
    print("Logistic Regression: ")
    print(best_grid)
    print(report)
    plot_cm(cm, title="Logistic Regression (with question length)")
    models['Logistic Regression'] = best_grid
    
    #Linear SVM    
    params = dict([
        ('clf__C', [0.01, 0.1, 1, 10, 100]),
        ('clf__loss', ['hinge', 'squared_hinge']),        
    ])

    pipe = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
                    ('tfidf', TfidfTransformer()),
                ])),
                ('length', FunctionTransformer(get_question_length, validate=False)),
            ])),
            ('clf', LinearSVC(class_weight='balanced', random_state=SEED))            
    ])
    
    best_f1 = 0
    best_grid = {}
    report = []
    cm = []
    for g in ParameterGrid(params):             
        pipe.set_params(**g)
        pipe.fit(train['question'], train['question_class'])
        preds = pipe.predict(dev['question'])
        p, r, f, s = precision_recall_fscore_support(dev['question_class'], preds, average='weighted')  
        print(g)
        print(f)
        if f > best_f1:
            best_f1 = f
            best_grid = g
            report = classification_report(dev['question_class'], preds)
            cm = confusion_matrix(dev['question_class'], preds)
    
    print("Linear SVM: ")
    print(best_grid)
    print(report)
    plot_cm(cm, title="Linear SVM (with question length)")
    models['Linear SVM'] = best_grid
        
    return models
    
if __name__ == '__main__':
    data = read_dataset_splits(reader=read_question_only_data)
    data = add_classes(data)
    models = run_with_question_length(data)