from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

from data_readers import read_question_only_data, read_dataset_splits
from config import Config

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

SEED = Config.SEED
random.seed(SEED)

def extend_question_class(time):
    if time < Config.THRESHOLD_SHORT:
        return Config.LABEL_SHORT
    elif time < Config.THRESHOLD_MEDIUM:
        return Config.LABEL_MEDIUM
    else:
        return Config.LABEL_LONG
    
def add_classes(data):
    '''
        Add label corresponding to question class.
        Also create a new column with text to make it easier for sklearn's
        CountVectorizer/TfidfTransformer API.
    '''
    for key, value in data.items():
        value['question_text'] = value.apply(lambda row: " ".join(row['question']), axis=1)
        value['question_class'] = value.apply(lambda row: 
            extend_question_class(row['response_time_sec']), axis=1)
        
    return data

def run_ridge_regression(data):
    ''' 
        Run ridge regression on raw training data.
        Probably not going to be used as we decided to make it a 
        classification problem.
    '''
    train, dev, test = data['train'], data['dev'], data['test']
    
    pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),    
            ('pipe', Ridge(random_state=SEED))            
    ])
        
    pipe.fit(train['question_text'], train['response_time_sec'])
    preds = pipe.predict(dev['question_text'])
    plt.plot(dev['response_time_sec'], preds, 'bx')
    print("Regression R2: {}".format(
            pipe.score(dev['question_text'], dev['response_time_sec'])))

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

    plt.savefig("cm_baseline_{}.png".format(title), dpi=300)
    plt.close()
    
def run_baselines(data):
    '''
        Input: Dictionary of data (tiny, train, dev, test)
        Runs three baselines: 
            1. Logistic Regression with balanced class weights
            2. SVM (Linear Kernel) with balanced class weights
            3. Dummy Classifier (stratified guessing strategy)
        
        Prints classification reports and saves confusion matrices as images.
        Note that F1 scores are WEIGHTED MACRO, and will be influenced
        by relative class weights.
        
        Grid search is done over regularization constant C, as well as 
        penalty (logistic regression) and type of loss function (SVM).
    '''
    train, dev, test = data['train'], data['dev'], data['test']    
    models = {}    

    #Logistic regression
    params = dict([
        ('clf__C', [0.01, 0.1, 1, 10, 100]),
        ('clf__class_weight', ['balanced']),
        ('clf__penalty', ['l2', 'l1']),
        ('clf__random_state', [SEED]),
    ])

    pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),    
            ('clf', LogisticRegression())            
    ])
    
    best_f1 = 0
    best_grid = {}
    report = []
    cm = []
    for g in ParameterGrid(params): 
        pipe.set_params(**g)
        pipe.fit(train['question_text'], train['question_class'])
        preds = pipe.predict(dev['question_text'])
        p, r, f, s = precision_recall_fscore_support(dev['question_class'], preds, average='weighted')  
        if f > best_f1:
            best_f1 = f
            best_grid = g
            report = classification_report(dev['question_class'], preds)
            cm = confusion_matrix(dev['question_class'], preds)
    
    print("Logistic Regression: ")
    print(best_grid)
    print(report)
    plot_cm(cm, title="Logistic Regression")
    models['Logistic Regression'] = best_grid
    
    #Linear SVM    
    params = dict([
        ('clf__C', [0.01, 0.1, 1, 10, 100]),
        ('clf__class_weight', ['balanced']),
        ('clf__loss', ['squared_hinge', 'hinge']),
        ('clf__random_state', [SEED]),
    ])

    pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),    
            ('clf', LinearSVC())            
    ])
    
    best_f1 = 0
    best_grid = {}
    report = []
    cm = []
    for g in ParameterGrid(params):             
        pipe.set_params(**g)
        pipe.fit(train['question_text'], train['question_class'])
        preds = pipe.predict(dev['question_text'])
        p, r, f, s = precision_recall_fscore_support(dev['question_class'], preds, average='weighted')  
        if f > best_f1:
            best_f1 = f
            best_grid = g
            report = classification_report(dev['question_class'], preds)
            cm = confusion_matrix(dev['question_class'], preds)
    
    print("Linear SVM: ")
    print(best_grid)
    print(report)
    plot_cm(cm, title="Linear SVM")
    models['Linear SVM'] = best_grid
    
    #Dummy
    pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),    
            ('clf', DummyClassifier(random_state=SEED))            
    ])
    
    pipe.fit(train['question_text'], train['question_class'])
    preds = pipe.predict(dev['question_text'])
    print("Dummy Classifier: ")
    report = classification_report(dev['question_class'], preds)
    print(report)
    cm = confusion_matrix(dev['question_class'], preds)
    plot_cm(cm, title="Dummy Classifier")
    
    return models
    
    #TODO: Error Analysis on baseline models.
    
if __name__ == '__main__':
    data = read_dataset_splits(reader=read_question_only_data)
    data = add_classes(data)
    models = run_baselines(data)