import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# veri yukleme
datas = pd.read_csv('files/veriler_tenis.csv')

# veri on isleme
print(datas)
# weather = datas[['outlook', 'windy', 'play']]
# print(weather)

# eksik veriler
# sci kit
# bkz: ml0002.py

# kategorik veriler
# polynominal kategorik verinin ayrı değişkenler gibi true - false gibi ayrı sütunlara dönüştürülmesi
# bkz: ml0003.py

# Sonuçların 1 dataframe'de toplanması
# dataframe'lerin birleştirilmesi
# bkz: ml0004.py

# Veri Kaynağını Bölme
# Boy, kilo ve yaştan cinsiyetin tahmin edilebilmesini istiyoruz.
# verilerin train ve test olarak ve
# bağımlı değişkeni (hedefi) ayırıyoruz. x : bağımsız değişkenler, y: bağımlı değişkenler
# bkz: ml0005.py

# 2.7 öz nitelik ölçekleme
# normalizasyon gibi
# bkz: ml0007.py

# Model İnşaası (Doğrusal Regresyon)
# görsel olarak da bakalım
# bkz: ml0009.py

# Doğru değikenleri mi alıyoruz? Tümünü almalı mıyız?
# Backward Elimination - Geri Eleme
# bkz: ml0010.py

##################################################################
# kategorik veriler



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
play = datas.iloc[:,-1:].values
# print(play)
play[:,0]=le.fit_transform(play[:,0])
# print(play)

windy = datas.iloc[:,-2:-1].values
# print(windy)
windy[:,0]=le.fit_transform(windy[:,0])
# print(windy)
#
#
outlooks = datas.iloc[:,0:1].values
# print(outlooks)
outlooks[:,0] = le.fit_transform(datas.iloc[:,0])
# print(outlooks)
# #
# # polynominal kategorik verinin ayrı değişkenler gibi true - false gibi ayrı sütunlara dönüştürülmesi
ohe = preprocessing.OneHotEncoder()
outlooks = ohe.fit_transform(outlooks).toarray()
# print(outlooks)
#
# # Sonuçların 1 dataframe'de toplanması
result = pd.DataFrame(data=outlooks, index=range(14), columns=['overcast','rainy','sunny'])
# print(result)
#
result2 = datas.iloc[:,1:3]
s = pd.concat([result,result2], axis=1)
print(s)

result3 =pd.DataFrame(data=windy, index=range(14), columns=['windy'])
result4 =pd.DataFrame(data=play, index=range(14), columns=['play'])
s2 = pd.concat([result3, result4], axis=1)
print(s2)

s3 = pd.concat([s,s2],axis=1)
print(s3)


# Veri Kaynağını Bölme
# Overcast, rainy, sunny hava durumlarından, sıcaklık, nem ve rüzgardan havanın tenis oynamaya uygun olup olmadığını tahmin etmek istiyoruz.
# verilerin train ve test olarak ve
# bağımlı değişkeni (hedefi) ayırıyoruz. x : bağımsız değişkenler, y: bağımlı değişkenler


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s3.iloc[:,:-1],s3.iloc[:,-1:],test_size=0.33,random_state=0)
# # veriyi 4'e böldük.
print(x_train)
print(y_train)
print(x_test)
print(y_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

print(tahmin)
print(y_test)

