import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer


# Veri setini yükle
df = pd.read_csv('files/happydata.csv')

# Özellikler ve hedef değişkeni ayırma
X = df.drop('happy', axis=1)
y = df['happy']

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test verisini %80-%20 oranında ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN modelini eğitmek için en iyi k değeri (örneğin k=5) kullanıyoruz
best_k = 8  # İstediğiniz k değeriyle değiştirebilirsiniz
# knn = KNeighborsClassifier(n_neighbors=best_k)
knn = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')
knn.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = knn.predict(X_test)

# Confusion Matrix hesapla
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Performans Metrikleri
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# Sonuçları yazdır
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall (Sensitivity): {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")


cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# **Specificity Hesaplama**:
# # True Negative (TN) ve False Positive (FP) hesaplayarak Specificity'yi buluyoruz
# TN = cm[0, 0]  # Gerçek negatifler
# FP = cm[0, 1]  # Yanlış pozitifler
# specificity = TN / (TN + FP)  # Specificity formülü
#
# print(f"Specificity: {specificity:.4f}")

# **ROC Eğrisini Çizme**
# Modelin probabilistik tahminlerini al
y_prob = knn.predict_proba(X_test)  # Probabilistik tahminler (sınıf 1'in olasılıkları)

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
plt.title('Receiver Operating Characteristic (ROC) Eğrisi \n Manhattan mesafesi KNN k=8')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
