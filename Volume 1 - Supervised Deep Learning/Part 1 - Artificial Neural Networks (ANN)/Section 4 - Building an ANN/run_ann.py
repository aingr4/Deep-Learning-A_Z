# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 17:33:58 2018

@author: aingr
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


classifier = ""
best_parameters = ""
best_accuracy = ""

class run_ann(object):
    def __init__(self, X_train, y_train, bs = 1, ep = 1, test = 0):
        
        
        if (test == 0):
            self.classifier = KerasClassifier(build_fn=self.build_classifier,
                                              batch_size = bs,
                                              epochs = ep)
            self.accuracies = cross_val_score(estimator=self.classifier,
                                              X  = X_train,
                                              y  = y_train,
                                              cv = 8)
        elif(test == 1):
            self.classifier = KerasClassifier(build_fn=self.build_classifier)
            self.classifier.fit(x = X_train, 
                                y = y_train, 
                                batch_size = bs, 
                                epochs = ep)
            
        elif(test == 2):
            self.classifier = KerasClassifier(build_fn=self.build_classifier_with_optimzer)
            self.parameters = {'batch_size': [25, 32],
                               'epochs': [500],
                               'optimizer': ['rmsprop'],
                               'units': [6,9,12],
                               'hidden_layers': [1,2,3]}
            
            grid_search = GridSearchCV(estimator = self.classifier,
                                       param_grid = self.parameters,
                                       cv = 10)
            
            grid_search = grid_search.fit(X_train, y_train)
            
            self.best_parameters = grid_search.best_params_
            self.best_accuracy = grid_search.best_score_
        
        
    def build_classifier(self):
        classifier = Sequential()

        classifier.add(Dense(units = 6, kernel_initializer='uniform',activation='relu'))
    
        classifier.add(Dense(units = 6, kernel_initializer='uniform',activation='relu'))
    
        classifier.add(Dense(units = 1, kernel_initializer='uniform',activation='sigmoid'))
    
        classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])    
        
        
        return classifier
    
    def build_classifier_with_dropout(self, dropout):
        classifier = Sequential()

        classifier.add(Dense(units = 6, kernel_initializer='uniform',activation='relu'))
        classifier.add(Dropout(rate = dropout))
        
        classifier.add(Dense(units = 6, kernel_initializer='uniform',activation='relu'))
        classifier.add(Dropout(rate = dropout))
        
        classifier.add(Dense(units = 1, kernel_initializer='uniform',activation='sigmoid'))
    
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])    
        
        return classifier
    
    def build_classifier_with_optimzer(self, optimizer, units, hidden_layers):
        classifier = Sequential()

        classifier.add(Dense(units = units, kernel_initializer='uniform',activation='relu'))
        
        for i in range(hidden_layers):
            classifier.add(Dense(units = units, kernel_initializer='uniform',activation='relu'))
            
        classifier.add(Dense(units = 1, kernel_initializer='uniform',activation='sigmoid'))
    
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])    
        
        return classifier
        
        
    def get_classifier(self):
        return self.classifier

    def get_best_params(self):
        return self.best_parameters
    
    def get_best_accuracy(self):
        return self.best_accuracy
        
        
        
        
        
        
        
        
        
        