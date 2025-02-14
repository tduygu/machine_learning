# 1. kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. veri önişleme
# 2.1 veri yükleme
datas = pd.read_csv('files/eksikveriler.csv')

print(datas)

# 2.2 eksik veriler
#sci kit
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
yas = datas.iloc[:,1:4].values

#ogren
imputer = imputer.fit(yas[:,1:4])
#ortalama deger olarak yerine koy
yas[:,1:4]=imputer.transform(yas[:,1:4])

# 2.3 encoding : kategorik veriler
countries = datas.iloc[:,0:1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(datas.iloc[:,0])


# polynominal kategorik verinin ayrı değişkenler gibi true - false gibi ayrı sütunlara dönüştürülmesi
ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()

# 2.4 Sonuçların 1 dataframe'de toplanması
result = pd.DataFrame(data=countries, index=range(22), columns=['fr','tr','us'])
result2 = pd.DataFrame(data=yas, index=range(22), columns=['boy', 'kilo', 'yas'])
cinsiyet = datas.iloc[:,-1].values
result3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])


# 2.5 dataframe'lerin birleştirilmesi
# s = pd.concat([result,result2])
s = pd.concat([result,result2], axis=1)
# print(s)

s2 = pd.concat([s, result3], axis=1)
# print(s2)

# 2.6 Veri Kaynağını Bölme
# Boy, kilo ve yaştan cinsiyetin tahmin edilebilmesini istiyoruz.
# verilerin train ve test olarak ve
# bağımlı değişkeni (hedefi) ayırıyoruz. x : bağımsız değişkenler, y: bağımlı değişkenler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, result3,test_size=0.33,random_state=0)
# veriyi 4'e böldük.
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

# 2.7 öz nitelik ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)