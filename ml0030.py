# Decision Tree - ID3 - decision tree çizimli
# odor çıkarıldı

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Entropi hesaplama fonksiyonu
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


# Bilgi kazancı hesaplama fonksiyonu
def information_gain(X, y, feature_index):
    total_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / sum(counts)) * entropy(y[X[:, feature_index] == v]) for i, v in enumerate(values))
    return total_entropy - weighted_entropy


# ID3 Karar Ağacı Düğümü
class Node:
    def __init__(self, feature=None, value=None, branches=None, label=None):
        self.feature = feature
        self.value = value
        self.branches = branches if branches is not None else {}
        self.label = label


# ID3 Algoritması
def id3(X, y, features):
    if len(set(y)) == 1:
        return Node(label=y[0])
    if len(features) == 0:
        return Node(label=Counter(y).most_common(1)[0][0])

    gains = [information_gain(X, y, f) for f in features]
    best_feature = features[np.argmax(gains)]

    node = Node(feature=best_feature)
    values = np.unique(X[:, best_feature])

    for value in values:
        sub_X = X[X[:, best_feature] == value]
        sub_y = y[X[:, best_feature] == value]
        if len(sub_y) == 0:
            continue
        new_features = [f for f in features if f != best_feature]
        node.branches[value] = id3(sub_X, sub_y, new_features)

    return node


# Karar ağacını yazdırma fonksiyonu
def print_tree(node, feature_names, indent=""):
    if node.label is not None:
        print(indent + "Leaf: " + str(node.label))
        return
    print(indent + f"Feature: {feature_names[node.feature]}")
    for value, subtree in node.branches.items():
        print(indent + f"- {value} ->")
        print_tree(subtree, feature_names, indent + "  ")


# Tahmin fonksiyonu
def predict(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature] in node.branches:
        return predict(node.branches[x[node.feature]], x)
    return 0  # Varsayılan sınıf


# Veri setini yükleme
data = pd.read_csv("files/mushrooms.csv")

# 'odor' özniteliğini çıkarma
data = data.drop(columns=['odor'])

# Kategorik verileri sayısal hale getir
label_encoders = {}
for col in data.columns:
    label_encoders[col] = {val: idx for idx, val in enumerate(data[col].unique())}
    data[col] = data[col].map(label_encoders[col])

# Özellikler ve hedef değişkeni ayırma
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
feature_names = data.columns[1:]

# Eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
features = list(range(X_train.shape[1]))
decision_tree = id3(X_train, y_train, features)

# Modeli değerlendirme
y_pred = [predict(decision_tree, x) for x in X_test]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Karar Ağacını Yazdırma
print("\nDecision Tree Structure:")
print_tree(decision_tree, feature_names)

# Performans Değerlendirmesi
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# ROC eğrisini çizme
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Karar Ağacı (Decision Tree) \n ID3 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zehirli Mantar Veri Seti için ROC Eğrisi')
plt.legend(loc='lower right')
plt.show()

# Model performansı değerlendirme
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

