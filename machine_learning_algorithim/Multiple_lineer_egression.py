import numpy as np
import pandas as pd
import   matplotlib.pyplot as plt
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression

evlilik =pd.read_csv("Student_Marks.csv")
x=evlilik.iloc[:,0:2].values  # multiple regression da 2 kolon birden x verisi olarak kullanıldığı için "reshape()" metodu kullanılmaz
y=evlilik.iloc[:,2].values.reshape(-1,1)
xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.3)
model=LinearRegression()
model.fit(xtrain,ytrain)
sonuc=model.predict([[15,13.98]])
print(sonuc)
