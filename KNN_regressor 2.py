#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:15:53 2022

@author: apple
"""

#import data set

import pandas as pd
dataset=pd.read_csv("/Users/apple/Desktop/RKCMS/50_Startups.csv")

#check if the null values

print(df.isnull().values.any())


#choose input and output
X=dataset.iloc[:,0:2].values
y=dataset.iloc[:,4].values


#trainind and testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)



#choose the model and find the accuracy fittinf nearest neighbour for regression
from sklearn.neighbors import KNeighborsRegressor
nnr=KNeighborsRegressor(n_neighbors=6)
nnr.fit(X_train,y_train)



print(nnr.score(X_test,y_test))

#find the value of K
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_val=[]
for K in range(1,20):
    nn_model=KNeighborsRegressor(n_neighbors=K)
    nn_model.fit(X_train,y_train)
    y_pred=nn_model.predict(X_test)
    rmse=sqrt(mean_squared_error(y_test,y_pred))
    rmse_val.append(rmse)
    print("RMSE value",rmse,"-----K",K)


#plot the graph for rmse

curve=pd.DataFrame(rmse_val)
curve.plot()








