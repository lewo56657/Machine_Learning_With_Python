import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

dataa=pd.read_csv("C:/Users/levent/OneDrive/Masaüstü/machine_learning_algorithim/sınıflandırma algoritmaları/urun.csv")
x=dataa[["yaş","maaş"]].values
y=dataa[["satinalma"]].values.reshape(-1,1)

xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.3,random_state=22)
sıfır=dataa[dataa.satinalma==0]
bir=dataa[dataa.satinalma==1]
sc=StandardScaler()
xtrain2=sc.fit_transform(xtrain)  
xtest2=sc.transform(xtest)
kn2=KNeighborsClassifier(n_neighbors=7)#knn algoritmasnın kütüphanesi import edilir ve içerisine en yakın kaç komşusu ile ilgileniyorsak eklenir.
kn2.fit(xtrain2,ytrain)
sonuc=kn2.predict(xtest2)
başarı=kn2.score(xtest2,ytest)#burada test edilecek xtest  ve ytest  değerlerinin birbrine yakınlık oranına yani başarı oranına bakılır.
cm=confusion_matrix(ytest,sonuc)#confisuan matrisi ile  ytest,sonuc bize doğru ve yanlış tahin sayılarını  değerlerin verilmesi sağlanır.
print(cm)

