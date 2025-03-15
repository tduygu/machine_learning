import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc


# Veri setini yükleme (Happiness Classification veri seti)
data = pd.read_csv("happiness_classification.csv")  # Gerçek dosya adını kullan

# Özellik ve hedef değişkenleri ayırma
X = data.drop(columns=['Happiness Score'])  # Tahmin edilmek istenen hedef değişken
y = data['Happiness Score']

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veri setini eğitim ve test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# KNN modelini oluştur ve eğit
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_resampled, y_resampled)

# Tahmin yap
y_pred = knn_model.predict(X_test)
y_prob = knn_model.predict_proba(X_test)[:, 1]  # Pozitif sınıfın olasılıkları

# Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# ROC eğrisini çizme
fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y_test.max())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'KNN (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN with SMOTE')
plt.legend(loc='lower right')
plt.show()

# Sonuçları yazdır
print(f'KNN Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
