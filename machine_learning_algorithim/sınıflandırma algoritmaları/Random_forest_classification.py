import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
import  seaborn as sea 


satış=pn.read_csv("C:/Users/levent/OneDrive/Masaüstü/machine_learning_algorithim/sınıflandırma algoritmaları/urun.csv")
x=satış[["yaş","maaş"]].values
y=satış[["satinalma"]].values.reshape(-1,1)
sıfır=satış[satış.satinalma==0]
bir=satış[satış.satinalma==1]
xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.3,random_state=22)
st=StandardScaler()
xtest2=st.fit_transform(xtest)
xtrain2=st.fit_transform(xtrain)
rf=RandomForestClassifier(n_estimators=100)
rf.fit(xtrain2,ytrain)
sonuc=rf.predict(xtest2)
başarı=rf.score(xtest2,ytest)
matris=confusion_matrix(ytest,sonuc)

#confission matrisini görselleştirmek için bu kod satırları kullanılabilir.
f,ax=plt.subplots(figsize=(5,5))
sea.heatmap(matris, annot=True,linewidths=0.5,linecolor="green",fmt=".0f",ax=ax)

plt.show()
