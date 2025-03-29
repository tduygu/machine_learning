import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("files/anxiety_depression_data.csv")

categorical_columns_with_na = [col for col in df.select_dtypes(include=['object']).columns if df[col].isnull().sum() > 0]

# Kategorik sütunlardaki eksik değerleri makine öğrenmesi ile doldurma
if categorical_columns_with_na:
    # data[categorical_columns_with_na] = data[categorical_columns_with_na].fillna("Bilinmiyor")
    imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_columns_with_na] = imputer.fit_transform(df[categorical_columns_with_na])

label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    print(col)
    df[col] = label_encoder.fit_transform(df[col])

# Korelasyon matrisini hesapla
corr_matrix = df.corr()

# Korelasyon matrisini görselleştir
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.show()



# Korelasyon matrisinin mutlak değerini al
corr_matrix_abs = corr_matrix.abs()

# Eşik değer belirleme (0.8 üzeri güçlü korelasyon)
threshold = 0.8

# Üst üçgen matrisini oluştur (çünkü korelasyon matrisi simetrik)
upper_triangle = corr_matrix_abs.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Belirlenen threshold'u aşan sütunları çıkar
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

print("Çıkarılacak öznitelikler:", to_drop)
#
# df.drop(columns=to_drop, inplace=True)
# print("Yeni veri seti şekli:", df.shape)
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
#
# # Bağımsız ve bağımlı değişkenleri ayırma
# X = df.drop(columns=['Satisfied'])  # Hedef değişken dışındaki tüm değişkenler
# y = df['Satisfied']
#
# # Eğitim ve test setlerine ayırma
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Modeli eğitme
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Tahmin yapma ve doğruluk hesaplama
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Yeni Modelin Doğruluk Oranı:", accuracy)

