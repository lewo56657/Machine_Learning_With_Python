import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

ürün=pn.read_csv("C:/Users/levent/OneDrive/Masaüstü/machine_learning_algorithim/sınıflandırma algoritmaları/urun.csv")
x=ürün[["yaş","maaş"]].values
y=ürün[["satinalma"]].values.reshape(-1,1)
sıfır=ürün[ürün.satinalma==0]
bir=ürün[ürün.satinalma==1]
xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.3,random_state=25)
sc=StandardScaler()
xtest2=sc.fit_transform(xtest)
xtrain2=sc.transform(xtrain)
destek=SVC(kernel="rbf")
destek.fit(xtrain2,ytrain)
sonuc=destek.predict(xtest2) 
başarı=destek.score(xtest2,ytest)
matris=confusion_matrix(ytest,sonuc)
#veri setinin etiketeine göre değerler sıralanddıktan sonra bu şekilde değerler ekrana gösterilir.
print(matris)





