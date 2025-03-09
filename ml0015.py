# Test ve Eğitim verisini bölerken K-Fold kullandığımda ne değişir?
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Veri setini yükle
# url = 'https://raw.githubusercontent.com/username/dataset-repo/main/happiness_classification.csv'
# df = pd.read_csv(url)
df = pd.read_csv('files/happydata.csv')
print(df)

# Özellikler ve hedef değişkeni ayırma
X = df.drop('happy', axis=1)
y = df['happy']
print(X)
print(y)


# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN modelini oluşturma
knn = KNeighborsClassifier(n_neighbors=5)

# K-Fold Cross-Validation (5 katlı)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation ile modelin başarısını değerlendirme
cv_scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')

# Sonuçları yazdırma
print("Cross-validation skorları:", cv_scores)
print("Ortalama doğruluk:", cv_scores.mean())
