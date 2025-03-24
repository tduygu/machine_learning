import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve


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
    def __init__(self, feature=None, value=None, branches=None, label=None, probability=None):
        self.feature = feature
        self.value = value
        self.branches = branches if branches is not None else {}
        self.label = label
        self.probability = probability


# ID3 Algoritması
def id3(X, y, features):
    if len(set(y)) == 1:
        return Node(label=y[0], probability=np.mean(y))
    if len(features) == 0:
        most_common_label = Counter(y).most_common(1)[0][0]
        return Node(label=most_common_label, probability=np.mean(y))

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
        print(indent + f"Leaf: {node.label} (P={node.probability:.2f})")
        return
    print(indent + f"Feature: {feature_names[node.feature]}")
    for value, subtree in node.branches.items():
        print(indent + f"- {value} ->")
        print_tree(subtree, feature_names, indent + "  ")


# Olasılık tahmini fonksiyonu
def predict_proba(node, x):
    if node.label is not None:
        return node.probability
    if x[node.feature] in node.branches:
        return predict_proba(node.branches[x[node.feature]], x)
    return 0.5  # Bilinmeyen durumlar için varsayılan olasılık


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
y_pred = [1 if predict_proba(decision_tree, x) >= 0.5 else 0 for x in X_test]
y_scores = [predict_proba(decision_tree, x) for x in X_test]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ROC Eğrisini Çizme
fpr, tpr, _ = roc_curve(y_test, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Rastgele tahmin çizgisi
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Karar Ağacını Yazdırma
print("\nDecision Tree Structure:")
print_tree(decision_tree, feature_names)
