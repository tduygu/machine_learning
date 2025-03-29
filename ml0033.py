# *** ödev 5 yapılan
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
data = pd.read_csv("files/anxiety_depression_data.csv")

# 'Life_Satisfaction_Score' özelliğini kategorik hale getirme
data['Life_Satisfaction_Score'] = data['Life_Satisfaction_Score'].apply(lambda x: 1 if x >= 7 else 0)
# data["Satisfied"] = (data["Life_Satisfaction_Score"] >= 7).astype(int)

# missing values var mı?
missing_values = data.isnull().sum()
print("Eksik değer sayıları:\n", missing_values[missing_values > 0])
# pd.set_option('display.max_columns', None)  # Tüm sütunları göster
# print(data.iloc[:100,:])
# print(data[["Medication_Use","Substance_Use"]])

# Sayısal sütunlardaki eksik değerleri ortalama ile doldurma
# df.fillna(df.mean(), inplace=True)

# Kategorik sütunlardaki eksik değerleri mod (en sık tekrar eden değer) ile doldurma
# categorical_columns = data.select_dtypes(include=['object']).columns
# data[categorical_columns] = data[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))
# df.select_dtypes(include=['object']).columns ifadesi, veri setindeki kategorik sütunları seçmek için kullanılıyor.
# df.select_dtypes(include=['number']).columns sayısal sütunları seçmek için
# df.select_dtypes(include=['int64']).columns int64 türündeki sütunları seçmek için vb.

# Kategorik sütunlardaki eksik değerleri makine öğrenmesi ile doldurma
# categorical_columns = data.select_dtypes(include=['object']).columns
# imputer = SimpleImputer(strategy="most_frequent")
# data[categorical_columns] = imputer.fit_transform(data[categorical_columns])
# Sadece eksik değer içeren kategorik sütunları seçme
categorical_columns_with_na = [col for col in data.select_dtypes(include=['object']).columns if data[col].isnull().sum() > 0]

# Kategorik sütunlardaki eksik değerleri makine öğrenmesi ile doldurma
if categorical_columns_with_na:
    # data[categorical_columns_with_na] = data[categorical_columns_with_na].fillna("Bilinmiyor")
    imputer = SimpleImputer(strategy="most_frequent")
    data[categorical_columns_with_na] = imputer.fit_transform(data[categorical_columns_with_na])


# veri dengeli mi?
# Hedef sütundaki sınıf dağılımını kontrol et
class_counts = data['Life_Satisfaction_Score'].value_counts()
# class_counts = data['Satisfied'].value_counts()
#
# # Yüzdelik oranlarını göster
class_percentages = data['Life_Satisfaction_Score'].value_counts(normalize=True) * 100
# class_percentages = data['Satisfied'].value_counts(normalize=True) * 100
# #
# Sonuçları yazdır
print("Sınıf Dağılımı:\n", class_counts)
print("\nSınıf Dağılımı (%)\n", class_percentages)


## Kategorik değişkenleri sayısal formata çevirme
# bu one hot encoding kullanıyor - desicion tree için label encoder daha iyi olabilir
# X = pd.get_dummies(X)
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    print(col)
    data[col] = label_encoder.fit_transform(data[col])

# Özellikler ve hedef değişkeni ayırma
# Hedef değişken (sondan ikinci sütun) Life_Satisfaction_Score
# y = data.iloc[:, -2]
y = data['Life_Satisfaction_Score']
# y = data['Satisfied']
# Bağımsız değişkenler (hedef değişken ve son sütun hariç tüm sütunlar)
# X = data.drop(columns=data.columns[-2])
# X = data.drop(['Medication_Use','Substance_Use','Life_Satisfaction_Score'], axis=1)
X = data.drop(['Life_Satisfaction_Score'], axis=1)



# Kontrol için boyutlar:
# print("X şekli:", X.shape)
# print("y şekli:", y.shape)
# print("data sekli:", data.shape)





# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)



# class_counts = y_resampled.value_counts()
# #
# # # Yüzdelik oranlarını göster
# class_percentages = y_resampled.value_counts(normalize=True) * 100
# # #
# # Sonuçları yazdır
# print("After Resampled")
# print("Sınıf Dağılımı:\n", class_counts)
# print("\nSınıf Dağılımı (%)\n", class_percentages)


# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Decision Tree Modeli (CART kullanarak)
# clf = DecisionTreeClassifier(criterion="gini")
# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(X_train, y_train)

# # # Modeli değerlendirme
y_pred = clf.predict(X_test)
# #

# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=10, criterion='gini')
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

#
#
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
# Karar ağacını çizdirme ????
# plt.figure(figsize=(20,10))
# plot_tree(clf, filled=True, feature_names=X.columns, class_names=["not satisfied","satisfied"], rounded=True)
# plt.show()


#
# Performans değerlendirmesi
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))
#
# # ROC eğrisini çizme
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# # ROC eğrisini çizme - RandomForest için
# fpr, tpr, thresholds = roc_curve(y_test, rfc.predict_proba(X_test)[:, 1])
# roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f'Karar Ağacı (Decision Tree) \n Gini (AUC = {roc_auc:.2f})')
plt.plot(fpr, tpr, color='blue', label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.title('Kaygı ve Depresyon Ruh Sağlığı Faktörleri Veri Seti için ROC Eğrisi')
plt.title('Kaygı ve Depresyon Ruh Sağlığı Faktörleri Veri Seti için ROC Eğrisi \n Random Forest')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()


