#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:26:50 2022

@author: apple
"""

import pandas as pd

dataset=pd.read_csv("/Users/apple/Desktop/RKCMS/Social_Network_Ads.csv")

X=dataset.iloc[:,[2,3]].values   #input for KNN model
y=dataset.iloc[:,4].values       #output for prediction


#Split into tain and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=3)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)


y_pred= knn.predict(X_test)


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred))
