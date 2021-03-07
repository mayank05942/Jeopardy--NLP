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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.corpus import stopwords
", ".join(stopwords.words('english'))

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from sklearn import preprocessing
#---------------------------------------------------------------------------
# Reading the modified dataset
df = pd.read_csv("/home/mayank/Desktop/mod_data.csv").dropna()

# Removing stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["ques"] = df["ques"].apply(lambda text: remove_stopwords(text))

# Label encoding $200 tp 0 and $1000 to 1

def label_encode(df):
    d = {"$200": 0,"$1000": 1}
    
    for i in range(len(df)):

        if df['value'][i] in d:
            df['value'][i] = d[df['value'][i]]
        else:
            df['value'][i] = 'None'
    return df
df = label_encode(df)



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
    

  

""" Different Deep Learning Models"""   


""" Model1: Feed Forward Neural Network"""

y_train_ff = keras.utils.to_categorical(y_train, num_classes=2) 
y_test_ff = keras.utils.to_categorical(y_test, num_classes=2)

model = Sequential()
dim = 3*len(meanlist)*300
model.add(Dense(1024, input_dim=dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


model.fit(training_data, y_train_ff,validation_split=0.1,shuffle=True,
          verbose=True,
          epochs=60,
          batch_size=256)

y_pred = model.predict(test_data, batch_size=256).argmax(axis=1)
y_test = np.array(y_test)
y_test = y_test.astype(np.int64)
print(classification_report(y_test, y_pred))    



