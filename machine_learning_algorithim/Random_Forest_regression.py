import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

bilet=pd.read_csv("bilet.csv")
x=bilet.sıra.values.reshape(-1,1)
y=bilet.fiyat.values.reshape(-1,1)
rf=RandomForestRegressor(n_estimators=100,random_state=25) #ormandan rastegele ağaçları seç ve karşılaştır demek 
# n_estimators=100 değeri nekadar ağaç karşılaştıracaağını seçer ,nekadar fazala olursa okadar doğru sonuç çıkar
rf.fit(x,y)
sınır=mp.arange(min(x),max(x),0.01).reshape(-1,1)
sonuc=rf.predict(sınır)
plt.scatter(sınır,y)

print(r2_score(y,sonuc)) #bizlere noktaların doğruya uaklığının ortalamassını ve hata ayını verir



