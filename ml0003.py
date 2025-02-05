import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
datas = pd.read_csv('files/veriler.csv')

# veri on isleme
print(datas)
print(datas[['yas']])

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

