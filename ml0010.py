import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
datas = pd.read_csv('files/veriler.csv')

# veri on isleme
print(datas)

# kategorik veriler
countries = datas.iloc[:,0:1].values
#print(countries)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(datas.iloc[:,0])
#print(countries)
#
# polynominal kategorik verinin ayrı değişkenler gibi true - false gibi ayrı sütunlara dönüştürülmesi
ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()
# print(countries)

# # kategorik veriler
cinsiyetler = datas.iloc[:,-1:].values
print(cinsiyetler)
# print(cinsiyetler[:,0])
# print(cinsiyetler[:,-1])
#
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cinsiyetler[:,0] = le.fit_transform(datas.iloc[:,-1])
print(cinsiyetler)

# polynominal kategorik verinin ayrı değişkenler gibi true - false gibi ayrı sütunlara dönüştürülmesi
ohe = preprocessing.OneHotEncoder()
cinsiyetler = ohe.fit_transform(cinsiyetler).toarray()
print(cinsiyetler)
#

#
# # Sonuçların 1 dataframe'de toplanması
result = pd.DataFrame(data=countries, index=range(22), columns=['fr','tr','us'])
#print(result)

yas = datas.iloc[:,1:4].values
result2 = pd.DataFrame(data=yas, index=range(22), columns=['boy', 'kilo', 'yas'])
#print(result2)

cinsiyet = datas.iloc[:,-1].values
#print(cinsiyet)
# result3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
# dummy variable - kukla değişken - cinsiyet için 2 kolon gereksiz, biri diğerini anlatıyor.
result3 = pd.DataFrame(data = cinsiyetler[:,:1], index=range(22), columns=['cinsiyet'])
print(result3)
#
# # dataframe'lerin birleştirilmesi
# s = pd.concat([result,result2])
s = pd.concat([result,result2], axis=1)
#print(s)

s2 = pd.concat([s, result3], axis=1)
print(s2)

# 2.6 Veri Kaynağını Bölme
# Boy, kilo ve yaştan cinsiyetin tahmin edilebilmesini istiyoruz.
# verilerin train ve test olarak ve
# bağımlı değişkeni (hedefi) ayırıyoruz. x : bağımsız değişkenler, y: bağımlı değişkenler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, result3,test_size=0.33,random_state=0)
# veriyi 4'e böldük.
print(x_train)
print(y_train)
print(x_test)
print(y_test)

# 2.7 öz nitelik ölçekleme
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)