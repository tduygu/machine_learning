# Bu kodda, K-Fold kullanarak veri setini 5 katmana bölecek ve her bir katmanda farklı k değerleri için
# modelin doğruluğunu hesaplayarak en iyi k değerini belirleyeceğiz.
#
#
# K-Fold Cross-Validation: Veri setini 5 katmana böler ve her katmanda eğitim ve test işlemi yaparak sonuçları ortalamalar.
# KNN Modeli: Farklı 𝑘 değerleriyle KNN modelini çalıştırır ve her 𝑘 için performans hesaplar.
# En İyi k: Hangi 𝑘?
# k değerinin en düşük hata oranına sahip olduğunu bulur.
#

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris  # Örnek veri seti, kendi veri setinizi buraya koyabilirsiniz
import matplotlib.pyplot as plt

df = pd.read_csv('files/happydata.csv')
X = df.drop('happy', axis=1)
y = df['happy']

# data = load_iris()
# X = data.data
# y = data.target

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN için farklı k değerleri deneyeceğiz
k_range = range(1, 21)  # 1 ile 20 arasındaki k değerlerini deniyoruz
error_rates = []  # Hata oranlarını saklamak için liste

# 5 katlı Cross-Validation kullanarak farklı k'lar için hata oranını hesapla
for k in k_range:
    # knn = KNeighborsClassifier(n_neighbors=k)
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

    # 5 katlı cross-validation ile doğruluk hesapla (scoring='accuracy' değil, 'neg_mean_squared_error' kullanacağız)
    # 'neg_mean_squared_error' hata oranı için negatif doğruluğu döndürür.
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

    # Hata oranını hesapla (negatif skorlar olduğu için negatif işareti kaldırıyoruz)
    error_rate = -np.mean(scores)
    error_rates.append(error_rate)

# En iyi k'yı bul
best_k = k_range[np.argmin(error_rates)]  # En düşük hata oranına sahip k
best_error_rate = np.min(error_rates)  # En düşük hata oranı

# Sonuçları yazdır
print(f"En iyi k değeri: {best_k}")
print(f"En iyi hata oranı: {best_error_rate:.4f}")

# Hata oranlarını çizdirme
plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker='o', linestyle='dashed', color='r')
plt.xticks(k_range)
plt.yticks(np.round(np.linspace(min(error_rates), max(error_rates), num=10), 4))
plt.xlabel("K Değeri")
plt.ylabel('Hata Oranı (Error Rate)')
# plt.title('KNN için K-Fold Cross-Validation ile \n KNN Algoritmasında En İyi K Değerini Bulma (Hata Oranı)')
plt.title('KNN için K-Fold Cross-Validation ile \n KNN Algoritmasında En İyi K Değerini Bulma (Hata Oranı) (Manhattan)')

plt.grid()
plt.show()

