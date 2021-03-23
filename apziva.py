# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:45:35 2021

@author: Casper
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
# %%
data=pd.read_csv("ACME.csv")
x=data.iloc[:,1:]
y=data.iloc[:,0].values.reshape(-1,1)
# %%
print(x.describe())
print(x.value_counts())
print(x.info())
# %%
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
x=ohe.fit_transform(x).toarray()

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

# %%
score_list=[]
for i in range(1,21,2):
    from sklearn.neighbors import KNeighborsClassifier
    knn_neighbors=i
    knn=KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn.fit(x_train,y_train)
    prediction=knn.predict(x_test)
    score_list.append(knn.score((x_test),(y_test)))

plt.plot(range(1,21,2),score_list)    
plt.xlabel("k values")    
plt.ylabel("accuary") 
 

# %%
knn=KNeighborsClassifier(n_neighbors=7)  
knn.fit(x_train,y_train)

knn1=KNeighborsClassifier(n_neighbors=7)
knn1.fit(x,y)

prediction=knn.predict(x)
cmx=confusion_matrix((y),(prediction))
print(cmx)

print("accuracy of all data:",knn1.score((x),(y)))
print("accuracy of seperated data:",knn.score((x_test),(y_test)))

# %%
import statsmodels.api as sm
X_list=data.iloc[:,[1,2,3,4,5,6]].values
r_ols=sm.OLS(y,X_list)
r=r_ols.fit()
print(r.summary())

