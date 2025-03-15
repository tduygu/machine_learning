# Naive Bayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize

# # Örnek veri setini yükleyelim (Iris veri seti)
# data = load_iris()
# X = data.data
# y = data.target
df = pd.read_csv('files/happydata.csv')

# Özellikler ve hedef değişkeni ayırma
X = df.drop('happy', axis=1)
y = df['happy']

# Veriyi ölçeklendir
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Sınıfları ikili formata dönüştürelim
# y_binarized = label_binarize(y, classes=[0, 1, 2])
y_binarized = label_binarize(y, classes=[0, 1])

# Veri setini eğitim ve test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gaussian Naive Bayes modelini oluştur ve eğit
model = GaussianNB()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# ROC eğrisini çizme
plt.figure()
for i in range(y_binarized.shape[1]):
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=[0, 1, 2])[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Sonuçları yazdır
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
