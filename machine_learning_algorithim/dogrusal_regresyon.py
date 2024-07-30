import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import  LinearRegression

ogrenci=pd.read_csv("Student_Marks.csv")
x=ogrenci.iloc[:,1].values.reshape(-1,1)
y=ogrenci.iloc[:,2].values.reshape(-1,1)
plt.scatter(x,y)
plt.xlabel("çalışma saati")
plt.ylabel("başarı puanı")

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3, random_state=33)
lr= LinearRegression()
lr.fit(xtrain,ytrain)
sonuc=lr.predict([[8]])
#plt.scatter(x,y)
#plt.plot(xtest,sonuc,color="red")
print(sonuc)