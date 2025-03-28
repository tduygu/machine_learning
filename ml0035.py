# *** ödev 5
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
data = pd.read_csv("files/mushrooms.csv")

# Kategorik verileri sayısal hale getir
label_encoders = {}
for col in data.columns:
    label_encoders[col] = {val: idx for idx, val in enumerate(data[col].unique())}
    data[col] = data[col].map(label_encoders[col])

# Özellikler ve hedef değişkeni ayırma
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

tree_rules = export_text(clf, feature_names=data.columns[1:])
print(tree_rules)
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=data.columns[1:], class_names=["edible", "poisonous"], filled=True)
plt.show()

y_prob = clf.predict_proba(X_test)[:, 1]
# ROC eğrisini çizme
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'gini (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - gini \n Mushroom Classification')
plt.legend(loc='lower right')
plt.show()