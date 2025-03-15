import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer


# İlgili diğer veri setleri
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data
# https://archive.ics.uci.edu/dataset/186/wine+quality
# https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data
df = pd.read_csv('files/winequality-red.csv')

# 'quality' özelliğini kategorik hale getirme
df['quality_category'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Güncellenmiş veri setini görüntüleme
print(df[['quality', 'quality_category']].head())
print(len(df))

# 'quality' sütununu kaldırma
df = df.drop(columns=['quality'])

# Güncellenmiş veri setini görüntüleme
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
print(df.iloc[:100,:])


# veri dengeli mi
# Hedef sütundaki sınıf dağılımını kontrol et
class_counts = df['quality_category'].value_counts()

# Yüzdelik oranlarını göster
class_percentages = df['quality_category'].value_counts(normalize=True) * 100
#
# Sonuçları yazdır
print("Sınıf Dağılımı:\n", class_counts)
print("\nSınıf Dağılımı (%)\n", class_percentages)


# Özellikler ve hedef değişkeni ayırma
X = df.drop('quality_category', axis=1)
y = df['quality_category']

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Elindeki veri setinde sadece sayısal öznitelikler varsa, normalizasyon gerekli değil.
# Çünkü Gaussian Naive Bayes (GNB) varsayılan olarak sayısal değişkenleri normal dağılıma göre işler.
# Ama normalizasyon yapmak performansı artırabilir.

# SMOTE uygulama
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Sonuç: Dengeleme Gerekli mi?
# Azınlık sınıfı %20 veya daha az ise → Dengeleme şart!
# Sınıflar nispeten dengeli ise → Gerekli olmayabilir.
# Naïve Bayes’te genellikle SMOTE veya ROS kullanmak önerilir.

# Yeni sınıf dağılımını kontrol etme
print(y_resampled.value_counts())

# Veriyi eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Naive Bayes modelini oluştur ve eğit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)[:, 1]

# Performans değerlendirmesi
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC eğrisini çizme
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Naïve Bayes (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naïve Bayes with Resampled Data')
plt.legend(loc='lower right')
plt.show()

# Model performansı değerlendirme
print(f'Naïve Bayes Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))