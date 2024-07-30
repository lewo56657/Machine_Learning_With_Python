import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

veriler=pd.read_csv("C:/Users/levent/OneDrive/Masaüstü/machine_learning_algorithim/sınıflandırma algoritmaları/urun.csv")
x=veriler[["yaş","maaş"]].values
y=veriler[["satinalma"]].values.reshape(-1,1)
xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.3,random_state=22)
sıfır=veriler[veriler.satinalma==0]
bir=veriler[veriler.satinalma==1]
sc=StandardScaler()
xtrain2=sc.fit_transform(xtrain)  
xtest2=sc.transform(xtest)
dc=DecisionTreeClassifier()
dc.fit(xtrain2,ytrain)
sonuc= dc.predict(xtest2)
başarı=dc.score(xtest2,ytest)
maatris=confusion_matrix(ytest,sonuc)
print(maatris )
print(başarı)

plt.show()