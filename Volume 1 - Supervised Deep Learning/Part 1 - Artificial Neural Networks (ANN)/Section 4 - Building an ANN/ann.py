# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:14:07 2018

@author: aingr
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('oh_enc', OneHotEncoder(sparse = False), [1,2])], remainder = 'passthrough')

X = ct.fit_transform(X)

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

####################################


# Making the Confusion Matrix


import time

import run_ann as rn
startTime = time.time()

run_ann = rn.run_ann(X_train, y_train,10,500,1)

elapsedTime = time.time() - startTime
print(elapsedTime)

params = run_ann.get_best_params()
accuracy = run_ann.get_best_accuracy()


#y_pred = run_ann.get_classifier().predict(X_test)
#y_pred = (y_pred > 0.5)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)










































