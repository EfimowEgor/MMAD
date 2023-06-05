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

def DTC(train):
    dtc_Model = DecisionTreeClassifier()
    dtc_Model.fit(train, train['fraud'])
    return dtc_Model

def RFC(train):
    rfc_Model = RandomForestClassifier()
    rfc_Model.fit(train, train['fraud'])
    return rfc_Model

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