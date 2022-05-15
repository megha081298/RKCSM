#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 13:07:14 2022

@author: apple
"""

import pandas as pd
dataset=pd.read_csv("Company_Performance.csv")
X=dataset.iloc[:,[0]].values
y=dataset.iloc[:,1].values

#fitting that into the liner regressor
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X,y)
print(linear_reg.score(X,y))


import matplotlib.pyplot as plt
plt.scatter(X,y,color="red")
plt.plot(X,linear_reg.predict(X),color="blue")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_ploy=poly_reg.fit_transform(X)

lin_reg_ploy=LinearRegression()
lin_reg_ploy.fit(X_ploy,y)
y_pred=lin_reg_ploy.predict(X_ploy)
print(lin_reg_ploy.score(X_ploy,y))

plt.scatter(X,y,color="red")
plt.plot(X,y_pred,color="blue")
plt.show()
