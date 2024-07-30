import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB # kontrol edeceğimiz etiket ondalıklı yani yüzde içeren bir kavramsa GaussianNB kullanılır.
#eğer evet,hayır yada alır,almaz gibi binary seçenekli ile BernoulliNB kullanılır.
#eğer etiket integer sayılardan ibaret ise MultinomialNB kullanılır

satış=pn.read_csv("C:/Users/levent/OneDrive/Masaüstü/machine_learning_algorithim/sınıflandırma algoritmaları/urun.csv")
x=satış[["yaş","maaş"]].values
y=satış[["satinalma"]].values.reshape(-1,1)
sıfır=satış[satış.satinalma==0]
bir=satış[satış.satinalma==1]
xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.3,random_state=25)
st=StandardScaler()
xtrain2=st.fit_transform(xtrain) #x değerleri firtransform edilir.
xtest2=st.fit_transform(xtest)
gs=GaussianNB()
br=BernoulliNB()
ml=MultinomialNB()
gs.fit(xtrain2,ytrain)
sonuc=gs.predict(xtest2)
brbaşarı=gs.score(xtest2,ytest)
matris=confusion_matrix(ytest,sonuc)
print(matris)



