# *** ödev 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Veri setini yükleme
data = pd.read_csv("files/happydata.csv")

# Son sütunu kategorik olarak dönüştürme
label_encoder = LabelEncoder()
data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])

# Özellikler ve hedef değişkeni ayırma
X = data.iloc[:, :-1].values  # Son sütun hedef değişken olarak alındı
y = data.iloc[:, -1].values

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Decision Tree Modeli (Entropy kullanarak ID3)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
### eklenen
# y_prob = clf.predict_proba(X_test)[:, 1]

# Karar Ağacını Çizme
plt.figure(figsize=(20,10))
# plot_tree(clf, feature_names=data.columns[:-1], class_names=['infoavail','housecost','schoolquality','policetrust','streetquality','events'], filled=True, label='root')
plot_tree(clf, feature_names=data.columns[:-1], class_names=['0','1'], filled=True, label='root') # buna bak

plt.show()

tree_rules = export_text(clf, feature_names=list(data.columns[:-1]))
print(tree_rules)

# Performans değerlendirmesi
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC eğrisini çizme
# burada y_prob kullanılmalıydı ?????
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Karar Ağacı (Decision Tree) \n C4.5 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mutluluk Veri Seti için ROC Eğrisi')
plt.legend(loc='lower right')
plt.show()


