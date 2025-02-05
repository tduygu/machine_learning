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


