import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
import seaborn as sns
from sklearn import neighbors

wine_dataset = pd.read_csv('winequality-red.csv')
wine_dataset_ = pd.read_csv('winequality-red.csv')

bins = [2,6.5,8]
group_names = ['bad','good']
wine_dataset['quality'] = pd.cut(wine_dataset['quality'],bins=bins,labels=group_names)
label_quality = LabelEncoder()
wine_dataset['quality'] = label_quality.fit_transform(wine_dataset['quality'])


X = wine_dataset.drop('quality', axis = 1)
Y = wine_dataset['quality']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def randomForestClassifier():
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train,Y_train)
    pred_rfc = rfc.predict(X_test)
    return ["Accuracy score for random forest: "  + str(100*accuracy_score(Y_test,pred_rfc)) +"%",rfc]
def KnearestNeighborsClassifier():
    knn = neighbors.KNeighborsClassifier(n_neighbors=10,p=2, weights='distance')
    knn.fit(X_train,Y_train)
    pred_knn = knn.predict(X_test)
    return ["Accuracy score for 10NN: "  + str(100*accuracy_score(Y_test,pred_knn)) +"%",knn]
def SVMClassifier():
    svc = SVC()
    svc.fit(X_train,Y_train)
    pred_svc = svc.predict(X_test)
    return ["Accuracy score for SVM: "  + str(100*accuracy_score(Y_test,pred_svc)) +"%",svc]

def gridSearchForSVM():
    param = {
        'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
        'kernel':['linear', 'rbf'],
        'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
    }
    grid_svc = GridSearchCV(SVMClassifier()[1], param_grid=param, scoring='accuracy', cv=10)
    grid_svc.fit(X_train,Y_train)
    para = grid_svc.best_params_
    svc2 = SVC(C = para['C'], gamma =  para['gamma'], kernel= para['kernel'])
    svc2.fit(X_train, Y_train)
    pred_svc2 = svc2.predict(X_test)
    return ["Accuracy score for SVM : "  + str(100*accuracy_score(Y_test,pred_svc2)) +"%",para]
def gridSearchForKNN():
    grid_params = {
        'n_neighbors':[8,10,15,20],
        'weights': ['uniform','distance'],
        'metric': ['euclidean','manhattan']

    }
    gs  = GridSearchCV(
        KnearestNeighborsClassifier()[1],
        grid_params,
        verbose=1,
        cv = 3,
        n_jobs=1
    )
    gs_res = gs.fit(X_train,Y_train)
    para = gs_res.best_params_
    knn2 = neighbors.KNeighborsClassifier(n_neighbors=para['n_neighbors'],weights=para['weights'])
    knn2.fit(X_train,Y_train)
    pred_knn2 = knn2.predict(X_test)
    return ["Accuracy score for KNN : "  + str(100*accuracy_score(Y_test,pred_knn2)) +"%",para]
def gridSearchForRandomForest():
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
    }
    CV_rfc = GridSearchCV(estimator=randomForestClassifier()[1], param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, Y_train)
    para = CV_rfc.best_params_
    rfc1 = RandomForestClassifier(random_state=42, max_features=para['max_features'], n_estimators= para['n_estimators'], max_depth=para['max_depth'], criterion=para['criterion'])
    rfc1.fit(X_train,Y_train)
    pred_rfc1=rfc1.predict(X_test)
    return ["Accuracy score for RFC : "  + str(100*accuracy_score(Y_test,pred_rfc1)) +"%",para]