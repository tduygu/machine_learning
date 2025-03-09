# KNN'de en iyi k değerini bulma

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv('files/happydata.csv')

# Özellikler ve hedef değişkeni ayırma
X = df.drop('happy', axis=1)
y = df['happy']

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim-Test bölme (isteğe bağlı)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Farklı K değerleri için hata oranlarını hesaplayalım
k_values = range(1, 21)
error_rates = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    # knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=3)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)  # Hata oranı = 1 - doğruluk oranı
    # error_rate = round(1 - accuracy_score(y_test, y_pred), 2)
    error_rates.append(error_rate)
print(f"K Değerleri İçin Hata Oranları: {error_rates}")

# 5. Hata oranlarına göre en iyi K değerini grafikte göster
plt.figure(figsize=(8, 5))

plt.plot(k_values, error_rates, marker='o', linestyle='solid', color='b')

plt.xticks(k_values)
plt.yticks(np.round(np.linspace(min(error_rates), max(error_rates), num=10), 4))
plt.xlabel("K Değeri")
plt.ylabel("Hata Oranı")
plt.title("K Değerine Göre Hata Oranı")
# plt.title("K Değerine Göre Hata Oranı - Manhattan Uzaklığı ")
# plt.title("K Değerine Göre Hata Oranı - Minkowski Uzaklığı ")

plt.grid()
plt.show()
