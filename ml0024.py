# Artık Gaussian yerine CategoricalNB kullanılıyor, çünkü tüm öznitelikler kategorik.
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.naive_bayes import CategoricalNB

df = pd.read_csv('files/mushrooms.csv')
print(df)

# 'stalk-root' sütununu kaldırma - eksik veri var. - yok.
# df = df.drop(columns=['stalk-root'])

# Kategorik verileri sayısal değerlere dönüştürme
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Güncellenmiş veri setini görüntüleme
# pd.set_option('display.max_columns', None)  # Tüm sütunları göster
print(df.iloc[:100,:])

#
# veri dengeli mi
# Hedef sütundaki sınıf dağılımını kontrol et
class_counts = df['class'].value_counts()

# Yüzdelik oranlarını göster
class_percentages = df['class'].value_counts(normalize=True) * 100
#
# Sonuçları yazdır
print("Sınıf Dağılımı:\n", class_counts)
print("\nSınıf Dağılımı (%)\n", class_percentages)
# Doğrudan etkileyen öznitelik var mı?
print("Sonucu doğrudan etkileyen öznitelik var mı?")
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
print(df.groupby("class").mean())

# Özellikler ve hedef değişkeni ayırma
X = df.drop('class', axis=1)
y = df['class']
#
# X = X.drop(columns=['veil-color'])  # veya gill-attachment
# # aralarında yüksek korelasyon var, en altta test edildi.

# # Veriyi eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Naive Bayes modelini oluştur ve eğit
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
#
# # Naive Bayes modelini oluştur ve eğit
nb = CategoricalNB()
nb.fit(X_train, y_train)
#
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:, 1]
# predict_proba(X_test) fonksiyonu, her sınıf için olasılık tahmini döndürür.
# [:, 1] ifadesi, pozitif sınıfın (1) olasılıklarını seçmek için kullanılır.
# Neden Gerekli?
#
# ROC Eğrisi (Receiver Operating Characteristic Curve) çizerken pozitif sınıfın olasılıklarına ihtiyaç duyulur.
# roc_curve(y_test, y_prob) fonksiyonu, ham tahminler yerine olasılık değerlerini ister.
# Eğer y_prob değişkeni kullanılmazsa, ROC eğrisi çizilemez ve AUC (Area Under Curve) hesaplanamaz.
# #

# Performans değerlendirmesi
print(f'Categorical Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix hesapla
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

#
# ROC eğrisini çizme
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Naive Bayes (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Kategorik Naive Bayes \n Mushroom Classification')
plt.legend(loc='lower right')
plt.show()
#

# # Bilgi sızıntısı olabilir mi?
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12,8))
# plt.title('Korelasyon Matrisi \n Özniteliklerin Hedef Değişkenle İlişkisi')
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.show()
