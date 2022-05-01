#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:40:20 2022

@author: apple
"""

import pandas as pd
df=pd.read_csv("pima-data.csv")

#Null values
print(df.isnull().values.any())

#3.Data Molding
dia_map={True:1,False:0}
df["diabetes"]=df["diabetes"].map(dia_map)

X=df.iloc[:,0:8]
y=df.iloc[:,-1]

 #splitting the dataset
 
 from sklearn.model_selection import train_test_split
 X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
 
 #5 Imputing 
 from sklearn.impute import SimpleImputer
 fill_0=SimpleImputer(missing_values=0,strategy="mean")
 X_train=fill_0.fit_transform(X_train)
 X_test=fill_0.transform(X_test)
 
 from sklearn.neighbors import KNeighborsClassifier
 knn=KNeighborsClassifier(n_neighbors=14)
 knn.fit(X_train,y_train)
 
 y_pred=knn.predict(X_test)
 print(y_pred)
 
 
