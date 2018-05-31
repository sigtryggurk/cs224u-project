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
import copy

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
    '''
    for key, value in data.items():
        value['question_class'] = value.apply(lambda row: 
            extend_question_class(row['response_time_sec']), axis=1)
        
    return data

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
    
def dummy_tokenizer(tokens):
    return tokens    
    
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
        ('clf__penalty', ['l2', 'l1']),        
    ])

    pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),    
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
    plot_cm(cm, title="Logistic Regression")
    models['Logistic Regression'] = best_grid
    
#    #Linear SVM    
    params = dict([
        ('clf__C', [0.01, 0.1, 1, 10, 100]),
        ('clf__loss', ['hinge', 'squared_hinge']),        
    ])

    pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),    
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
    plot_cm(cm, title="Linear SVM")
    models['Linear SVM'] = best_grid
    
    #Dummy
    pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),    
            ('clf', DummyClassifier(random_state=SEED))            
    ])
    
    pipe.fit(train['question'], train['question_class'])
    preds = pipe.predict(dev['question'])
    print("Dummy Classifier: ")
    report = classification_report(dev['question_class'], preds)
    print(report)
    cm = confusion_matrix(dev['question_class'], preds)
    plot_cm(cm, title="Dummy Classifier")
    
    return models
    
if __name__ == '__main__':
    data = read_dataset_splits(reader=read_question_only_data)
    data = add_classes(data)
    models = run_baselines(data)
    
    train, dev, test = data['train'], data['dev'], data['test']    
    dev_new = copy.deepcopy(dev)
    pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=dummy_tokenizer, lowercase=False)),
            ('tfidf', TfidfTransformer()),    
            ('clf', LogisticRegression(class_weight='balanced', random_seed=SEED))            
    ])
    
    #Generate results for Logistic regression and output to CSV file.
    #True class, predicted class and class probabilities.
    g = models['Logistic Regression']
    pipe.set_params(**g)
    pipe.fit(train['question'], train['question_class'])
    dev_new['predicted_class'] = pipe.predict(dev_new['question'])
    probs = pipe.predict_proba(dev_new['question'])
    dev_new['prob_long'] = probs[:,0]
    dev_new['prob_medium'] = probs[:,1]
    dev_new['prob_short'] = probs[:,2]
    dev_new = dev_new.drop(['question'], axis=1)
    dev_new.to_csv(Config.BASELINE_PREDS_FILE)