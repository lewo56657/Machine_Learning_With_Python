import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing  import PolynomialFeatures,StandardScaler
from sklearn.svm import SVR


maaş=pn.read_csv("maaslar.csv")
print(maaş)

x=maaş[["Egitim Seviyesi"]].values.reshape(-1,1)
y=maaş[["maas"]].values.reshape(-1,1)
#burada standartlaştırma önemli olduğu için öncelikle x ve y değerleri StandardScaler() fonksiyonu ile standartlaştırılır. 
sc=StandardScaler()
x1=sc.fit_transform(x)
y1=sc.fit_transform(y)
sv=SVR(kernel="rbf")
sv.fit(x1,y1)
sonuc=sv.predict(x1)
plt.scatter(x1,y1)
plt.plot(x1,sonuc,color="red")
plt.show()
