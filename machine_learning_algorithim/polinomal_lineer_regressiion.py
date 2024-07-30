import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures # burada PolynomialFeatures kütüphanesi import edilir.

seviye=pn.read_csv("rank_salary.csv")

x=seviye.Level.values.reshape(-1,1)
y=seviye.Salary.values.reshape(-1,1)


pr=PolynomialFeatures(degree=4) #kütüphanenin bir bi nesnesi üretilir.
# diğer regressionlardan farklı olarak 2 veya 3 kuvvetli x değerleri ile çalışmamız gerektiği için x değerlerini   ,
# fit_transform(x) fonksiyonu ile dönüştürmemiz gerekir.
xpl=pr.fit_transform(x)
lr2=LinearRegression()
lr2.fit(xpl,y)
sonuc=lr2.predict(xpl)

plt.scatter(x,y)
plt.plot(x,sonuc,color="red")
plt.show()

