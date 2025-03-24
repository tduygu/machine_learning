# Decision Tree - ID3
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


# Tahmin fonksiyonu
def predict(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature] in node.branches:
        return predict(node.branches[x[node.feature]], x)
    return 0  # Varsayılan sınıf


# Veri setini yükleme
data = pd.read_csv("files/mushrooms.csv")

# Kategorik verileri sayısal hale getir
label_encoders = {}
for col in data.columns:
    label_encoders[col] = {val: idx for idx, val in enumerate(data[col].unique())}
    data[col] = data[col].map(label_encoders[col])

# Özellikler ve hedef değişkeni ayırma
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

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
