import re
import fasttext
import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.model_selection import train_test_split
from fasttext import load_model

import sys
sys.path.append('/home/mayank/Downloads/pmeans')
import p_mean_FT as pmeanFT
meanlist=['mean','p_mean_2','p_mean_3']
#meanlist=['mean','min','max']
#meanlist = ['mean']
import sklearn
from sklearn.metrics import classification_report
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
from xgboost import XGBClassifier
#---------------------------------------------------------------------------
# Reading the modified dataset
df = pd.read_csv("/home/mayank/Desktop/mod_data.csv").dropna()

# Removing stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["ques"] = df["ques"].apply(lambda text: remove_stopwords(text))

# Dividing into training and testing data
y = df.pop('value')
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

train_ques = X_train["ques"]
train_cat = X_train["category"]
train_ans = X_train["ans"]

test_ques = X_test["ques"]
test_cat = X_test["category"]
test_ans = X_test["ans"]

""" Using Fasttext Models trained prev generating word embeddings using power means concetenation approach """

def create_feature_matrix(data,model):
  # data will be X_train and X_test
  """ data input -> will be in the form on sentences "hello world"
      Step1 -> format like -> ["hello","world"]
      step2-> using pmean create a feature matrix """
  temp = []
  for items in data:
    temp.append(list(items.split()))

  feature_matrix = []
  for sentences in temp:
    feature_matrix.append(pmeanFT.get_sentence_embedding(sentences, model,meanlist))
  
  return feature_matrix

# Loading our Fastext Models

model_ques = fasttext.load_model('/home/mayank/Desktop/ques_model.bin')
model_ans = fasttext.load_model('/home/mayank/Desktop/ans_model.bin')
model_cat = fasttext.load_model('/home/mayank/Desktop/cat_model.bin')

# Generate Feature matrix: each senetence have dim= 300

train_ques_vec = create_feature_matrix(train_ques,model_ques)
train_ans_vec = create_feature_matrix(train_ans,model_ans)
train_cat_vec = create_feature_matrix(train_cat,model_cat)

test_ques_vec = create_feature_matrix(test_ques,model_ques)
test_ans_vec = create_feature_matrix(test_ans,model_ans)
test_cat_vec = create_feature_matrix(test_cat,model_cat)

train_ques_vec = np.array(train_ques_vec)
train_ans_vec = np.array(train_ans_vec)
train_cat_vec = np.array(train_cat_vec)

test_ques_vec = np.array(test_ques_vec)
test_ans_vec = np.array(test_ans_vec)
test_cat_vec = np.array(test_cat_vec)


"""" Concetenating the vectors of ques,ans and category into a single vector for both testing and training data"""

train1 =[]
training_data = []
for i in range(len(train_ques_vec)):
    train1.append((train_ques_vec[i],train_ans_vec[i],train_cat_vec[i]))
for i in range(len(train_ques_vec)):
    x,y,z=train1[i]
    training_data.append(np.concatenate([x,y,z], axis=0))
    
test1 =[]
test_data = []
for i in range(len(test_ques_vec)):
    test1.append((test_ques_vec[i],test_ans_vec[i],test_cat_vec[i]))
for i in range(len(test_ques_vec)):
    x,y,z=test1[i]
    test_data.append(np.concatenate([x,y,z], axis=0))

training_data = np.array(training_data)
test_data = np.array(test_data)
    

  

""" Different ML Models with grid search to optimize hyperparameters"""   


""" Model1: SVM Classifier """

# def train_classifier(X,y):
    
#     """ To perform grid search"""
#     param_grid = {'C': [0.1, 1, 10,100,1000],  
#                 'gamma': [1, 0.1, 0.01], 
#                 'kernel': ['rbf']}  
  
#     clf = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
#     clf.fit(X,y)
#     return clf

# classifier = train_classifier(training_data,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(test_data,y_test))
# print(classifier.best_params_)
    
# SVM = svm.SVC(kernel='rbf',C=0.1,gamma = 1)
# SVM.fit(training_data,y_train)

# pred_y = SVM.predict(test_data)
# print(sklearn.metrics.classification_report(y_test, pred_y))


""" Model2: Random Forest Classifier"""

# def train_classifier(X,y):
    
#     rfc=RandomForestClassifier(random_state=42)
#     param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
#     }
#     clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5,refit=True)
#     clf.fit(X,y)
#     return clf
# classifier = train_classifier(training_data,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(test_data,y_test))
# print(classifier.best_params_)

rfc =RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=4, criterion='gini')
rfc.fit(training_data,y_train)
pred_y = rfc.predict(test_data)
print(sklearn.metrics.classification_report(y_test, pred_y))


""" Model3: XG Boost Classifier"""


# def train_classifier(X,y):
#     """ To perform grid search"""
#     estimator = XGBClassifier(
#     objective= 'binary:logistic' ,
#     nthread=4,
#     seed=42
#     )
#     parameters = {
#         'max_depth': range (2, 10, 1),
#         'n_estimators': range(60, 220, 40),
#         'learning_rate': [0.1, 0.01, 0.05]
#     }
#     clf = GridSearchCV(
#         estimator=estimator,
#         param_grid=parameters,
#         scoring = 'accuracy',
#         n_jobs = -1,
#         cv = 10,
#         verbose=True
#     )
#     clf.fit(X, y)
#     return clf
# classifier = train_classifier(training_data,y_train)
# print (classifier.best_score_, "---> Best Accuracy score on Cross Validation Sets")
# print (classifier.score(test_data,y_test))
# print(classifier.best_params_)

# xgb = XGBClassifier(learning_rate = 0.1,max_depth=2,n_estimators= 140,objective="binary:logistic")
# xgb.fit(training_data,y_train)
# pred_y = xgb.predict(test_data)
# print(classification_report(y_test, pred_y))


