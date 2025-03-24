# Decision Tree - Entropi temelli ID3 Algoritması - sklearn ile
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_text

# Veri setini yükleme ve işleme
data = pd.read_csv("files/mushrooms.csv")
data = data.apply(lambda x: pd.factorize(x)[0])  # Kategorik verileri sayısal hale getir

# Özellikler ve hedef değişkeni ayırma
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ID3 Algoritması (Entropy kullanarak)
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=data.columns[1:], class_names=["edible", "poisonous"], filled=True)
plt.show()

tree_rules = export_text(clf, feature_names=list(data.columns[1:]))
print(tree_rules)


# Performans değerlendirmesi
print(f'Karar Ağacı - ID3 Accuracy: {accuracy_score(y_test, y_pred):.4f}')
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
# # ROC eğrisini çizme
# fpr, tpr, _ = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
#
# plt.figure()
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'Decision Tree - Entropy (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve - Decision Tree - Entropy \n Mushroom Classification')
# plt.legend(loc='lower right')
# plt.show()


