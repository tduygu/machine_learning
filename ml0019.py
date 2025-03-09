# Tümü
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris  # Örnek veri seti, kendi veri setinizi buraya koyabilirsiniz
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Veri setini yükle
df = pd.read_csv('files/happydata.csv')

# Özellikler ve hedef değişkeni ayırma
X = df.drop('happy', axis=1)
y = df['happy']

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN için farklı k değerleri deneyeceğiz
k_range = range(1, 21)  # 1 ile 20 arasındaki k değerlerini deniyoruz
error_rates = []  # Hata oranlarını saklamak için liste

# 5 katlı Cross-Validation kullanarak farklı k'lar için hata oranını hesapla
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

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
plt.xlabel('K Değeri')
plt.ylabel('Hata Oranı (Error Rate)')
plt.title('KNN için K-Fold Cross-Validation ile En İyi K Değerini Bulma (Hata Oranı)')
plt.grid(True)
plt.show()

# **ROC Eğrisini Çizme**
# Eğitim ve test verisi oluştur
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# En iyi k'yi kullanarak KNN modelini eğit
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_prob = best_knn.predict_proba(X_test)  # Probabilistik tahminler (sınıf 1'in olasılıkları)

# ROC eğrisini çizmek için her sınıfın pozitif sınıfını alıyoruz
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)  # Pos_label'ı her sınıf için değiştirebilirsiniz

# ROC eğrisinin AUC'sini hesapla
roc_auc = auc(fpr, tpr)

# ROC Eğrisini çizdir
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Rastgele tahmin çizgisi (diagonal)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Eğrisi')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

