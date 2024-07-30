import pandas as pn
import numpy as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.preprocessing  import PolynomialFeatures,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


data=pn.read_csv("bilet.csv")
print(data)
x=data.sıra.values.reshape(-1,1)
y=data.fiyat.values.reshape(-1,1)
dt=DecisionTreeRegressor()
dt.fit(x,y)
arr=mp.arange(min(x),max(x),0.1).reshape(-1,1)  #burada doğrusal değil bölmeli bir yapı elde etmek için x i bir numpy dizisie
#daha sık aralıklar ile ekleyip devamında bu diziyi kullanırız 
sonuc=dt.predict(arr)
plt.scatter(x,y)
plt.plot(arr,sonuc,color="red")
plt.show()



