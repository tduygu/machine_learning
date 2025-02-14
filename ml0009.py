# 1. kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. veri önişleme
# 2.1 veri yükleme
datas = pd.read_csv('files/AylaraGoreSatis.csv')

# print(datas)

aylar = datas[['Aylar']]
satislar = datas[['Satislar']]
# print(aylar)
# print(satislar)

# satislar2 = datas.iloc[:,1:].values
# print(satislar2)

# Veri Kaynağını Bölme
# Aylar : bağımsız değişken, Satislar: bağımlı değişken
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar,test_size=0.33,random_state=0)
# veriyi 4'e böldük.
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)
#
# # 2.7 öz nitelik ölçekleme
'''from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# 3. Model İnşaası (Doğrusal Regresyon)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
tahmin = lr.predict(X_test)

print(tahmin)
print(Y_test)
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

print(tahmin)
print(y_test)

# görsel olarak bakalım:
# önce sıralayalım: indexe göre sırala

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, tahmin)
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show()