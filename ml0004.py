import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
datas = pd.read_csv('files/eksikveriler.csv')

# veri on isleme
print(datas)
print(datas[['yas']])

# eksik veriler
#sci kit
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
yas = datas.iloc[:,1:4].values
print(yas)
#ogren
imputer = imputer.fit(yas[:,1:4])
#ortalama deger olarak yerine koy
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

# kategorik veriler

countries = datas.iloc[:,0:1].values
print(countries)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(datas.iloc[:,0])
print(countries)

# polynominal kategorik verinin ayrı değişkenler gibi true - false gibi ayrı sütunlara dönüştürülmesi
ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()
print(countries)

# Sonuçların 1 dataframe'de toplanması
result = pd.DataFrame(data=countries, index=range(22), columns=['fr','tr','us'])
print(result)

result2 = pd.DataFrame(data=yas, index=range(22), columns=['boy', 'kilo', 'yas'])
print(result2)

cinsiyet = datas.iloc[:,-1].values
print(cinsiyet)
result3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(result3)

# dataframe'lerin birleştirilmesi
# s = pd.concat([result,result2])
s = pd.concat([result,result2], axis=1)
print(s)

s2 = pd.concat([s, result3], axis=1)
print(s2)