import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri setini yükle
datas = pd.read_csv('files/happydata.csv')
df = pd.read_csv('files/happydata.csv')
X = df.drop('happy', axis=1)
y = df['happy']
# print(datas)

# Veri ön işleme
# print(datas[['infoavail','housecost']])
happy = datas.iloc[:,-1].values
dependent_data = pd.DataFrame(data=happy, index=range(143), columns=['happy'])
# print(dependent_data)

# Veri dengeli mi?
# print(sum(happy)) # 1'lerin sayısı
# ones = 0
# zeros = 0
# for i in range(143):
#     if int(happy[i]) == 0:
#         zeros += 1
#     else:
#         ones += 1
# print(f"ones: {ones}")
# print(f"zeros: {zeros}")
# Hedef sütundaki sınıf dağılımını kontrol et
class_counts = df['happy'].value_counts()

# Yüzdelik oranlarını göster
class_percentages = df['happy'].value_counts(normalize=True) * 100

# Sonuçları yazdır
print("Sınıf Dağılımı:\n", class_counts)
print("\nSınıf Dağılımı (%)\n", class_percentages)


x_data = datas.iloc[:,:-1]
independent_data = pd.DataFrame(data=x_data, index=range(143), columns=['infoavail', 'housecost', 'schoolquality', 'policetrust', 'streetquality', 'events'])
# print(independent_data)


# 2. Eğitim ve test veri setine ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(independent_data, dependent_data, test_size=0.2, random_state=42, stratify=y)

# print(f"x TEST \n {X_test}")
# print(f"y TEST \n {y_test}")


# 3. Veriyi ölçeklendirme (KNN, mesafeye dayalı çalıştığı için önemli!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# print(f"Ölçeklendirilmiş x TEST \n {X_test}")


# 4. KNN modelini oluştur ve eğit
k = 4  # Komşu sayısı (deneme yanılma ile en iyi değeri bulabilirsin)
knn = KNeighborsClassifier(n_neighbors=k)
# knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
y_train = np.ravel(y_train) # DataConversionWarning: A column-vector y was passed when a 1d array was expected. uyarısı için yapıldı
knn.fit(X_train, y_train)


# 5. Tahmin yap ve doğruluk hesapla
y_pred = knn.predict(X_test)
print(np.ravel(y_test))
print(y_pred)
# y_test_df = pd.DataFrame(data=y_test, index=range(29), columns=['y_test'])
# y_pred_df = pd.DataFrame(data=y_pred, index=range(29), columns=['y_pred'])
# y_test_pred = pd.concat([y_test_df,y_pred_df], axis=1)
# print(y_test_pred)
# print(y_test)
# print(y_pred)


accuracy = accuracy_score(y_test, y_pred)

# 6. Sonuçları yazdır
print(f"KNN Doğruluk Oranı: {accuracy:.2f}")
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
#

# 4. Farklı K değerleri için hata oranlarını hesaplayalım
# k_values = range(1, 21)
# error_rates = []  # Hata oranlarını saklayacak liste
#
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     # knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     # error_rate = 1 - accuracy_score(y_test, y_pred)  # Hata oranı = 1 - doğruluk oranı
#     error_rate = round(1 - accuracy_score(y_test, y_pred), 2)
#     error_rates.append(error_rate)
# print(f"K Değerleri İçin Hata Oranları: {error_rates}")
#
# # 5. Hata oranlarına göre en iyi K değerini grafikte göster
# plt.plot(k_values, error_rates, marker='o', linestyle='solid', color='b')
# plt.xlabel("K Değeri")
# plt.ylabel("Hata Oranı")
# plt.title("K Değerine Göre Hata Oranı")
# plt.show()


