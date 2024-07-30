import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import StandardScaler
dataa=pd.read_csv("C:/Users/levent/OneDrive/Masaüstü/machine_learning_algorithim/sınıflandırma algoritmaları/urun.csv")
x=dataa[["yaş","maaş"]].values
y=dataa[["satinalma"]].values.reshape(-1,1)

xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.3,random_state=22)
sıfır=dataa[dataa.satinalma==0]
bir=dataa[dataa.satinalma==1]
st=StandardScaler()
xtr2=st.fit_transform(xtrain) #x değerleri firtransform edilir.
xte2=st.fit_transform(xtest)
lr=LinearRegression()
lr.fit(xtr2,ytrain)
sonuc=lr.predict(xte2)
skor=lr.score(xtest,ytest)
print(skor)



 
