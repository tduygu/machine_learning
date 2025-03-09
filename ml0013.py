import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Wine veri setini yükle
wine = load_wine()
X, y = wine.data, wine.target
print(X)
print(y)

# 2. Eğitim ve test veri setine ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Veriyi ölçeklendirme (KNN, mesafeye dayalı çalıştığı için önemli!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. KNN modelini oluştur ve eğit
k = 5  # Komşu sayısı (deneme yanılma ile en iyi değeri bulabilirsin)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 5. Tahmin yap ve doğruluk hesapla
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 6. Sonuçları yazdır
print(f"KNN Doğruluk Oranı: {accuracy:.2f}")
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# 7. En iyi K değerini bulmak için test edelim
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# 8. En iyi K değerini grafikte göster
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b')
plt.xlabel("K Değeri")
plt.ylabel("Doğruluk Oranı")
plt.title("K Değerine Göre Doğruluk Oranı")
plt.show()



