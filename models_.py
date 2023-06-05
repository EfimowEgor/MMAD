import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def _SVC(train):
    svc_Model = SVC()
    svc_Model.fit(train, train['fraud'])
    return svc_Model

def Standard_Scaler(train):
    scaler = StandardScaler()
    scaler.fit(train.drop('fraud', axis=1))
    scaled_features = scaler.transform(train.drop('fraud', axis=1))
    scaled_data = pd.DataFrame(scaled_features, columns = train.drop('fraud', axis=1).columns)
    return scaled_data

def Logistic_Regression(train):
    x = Standard_Scaler(train)
    y = train['fraud']
    x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)
    return x_training_data, x_test_data, y_training_data, y_test_data

def kNN(x_training_data, y_training_data):
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(x_training_data, y_training_data)
    return model

def FCNN():
    model = Sequential()
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def XGB():
    param_grid = {
                    'learning_rate': [0.1, 0.01, 0.001, 0.03, 0.3, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'n_estimators': [10, 50, 100, 150, 200, 250, 300, 400, 500]
                 }
    model = XGBClassifier(n_estimators=100, objective='binary:logistic')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2)
    return grid_search

def RFC():
    param_grid = {
                    'n_estimators': [10, 30, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 10000],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 12, 15]
                 }
    model = RandomForestClassifier(n_estimators=100)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2)
    return grid_search

def CAT():
    param_grid = {
                    'iterations':[5,10,50,100],
                    'learning_rate':[0.01, 0.03, 0.1,1.0],
                    'bootstrap_type':['Bayesian', 'Bernoulli', 'MVS', 'No']
                 }
    model = CatBoostClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2)
    return grid_search