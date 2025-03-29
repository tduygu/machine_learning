import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Veri setini yükleme
df = pd.read_csv("files/anxiety_depression_data.csv")


# Hedef değişkeni kategorik hale getirme (>=7: Satisfied, <7: Unsatisfied)
df["Satisfied"] = (df["Life_Satisfaction_Score"] >= 7).astype(int)
df.drop(columns=["Life_Satisfaction_Score"], inplace=True)

# Eksik verileri doldurma
df.fillna(df.mode().iloc[0], inplace=True)

# Kategorik değişkenleri sayısal hale getirme
df = pd.get_dummies(df)





# Bağımsız ve bağımlı değişkenleri ayırma
X = df.drop(columns=["Satisfied"])
y = df["Satisfied"]


import numpy as np

# Korelasyon matrisini hesapla
corr_matrix = X.corr().abs()

# Üst üçgen matrisini oluştur
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Eşik değeri belirle (0.8'den büyük olanları çıkar)
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

print("Çıkarılacak öznitelikler:", to_drop)

# Veri setinden gereksiz sütunları çıkar
X_selected = X.drop(columns=to_drop)


# Veriyi eğitim ve test setlerine ayırma
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)



from sklearn.model_selection import GridSearchCV

# Hiperparametre aralıklarını belirleme
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

# GridSearchCV ile en iyi parametreleri bul
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi parametrelerle modeli yeniden eğitme
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Test setinde tahmin yapma
y_pred = best_model.predict(X_test)

# Yeni doğruluk oranı
accuracy = accuracy_score(y_test, y_pred)
print("En İyi Modelin Doğruluk Oranı:", accuracy)
print("En İyi Parametreler:", grid_search.best_params_)








#
# # Gini ile Decision Tree Modeli
# model = DecisionTreeClassifier(criterion="gini", random_state=42)
# model.fit(X_train, y_train)
#
# # Tahmin yapma ve doğruluk ölçme
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
#
# print(f"Mental Health Dataset Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Pozitif sınıfı belirle (Satisfied = 1)
# y_scores = model.predict_proba(X_test)[:, 1]
y_scores = best_model.predict_proba(X_test)[:, 1]

# ROC Curve hesaplama
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# ROC Eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")  # Rastgele sınıflandırma çizgisi
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Mental Health Dataset")
plt.legend(loc="lower right")
plt.show()

