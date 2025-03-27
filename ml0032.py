# analyticsvidhya
# https://www.analyticsvidhya.com/blog/2021/07/a-comprehensive-guide-to-decision-trees/
from markdown_it.rules_core import inline
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data[:,2:]
y = iris.target

# print(iris)
# print(X)
# print(y)

tree_classifier = DecisionTreeClassifier(max_depth=2)
tree_classifier.fit(X,y)

txt_tree =export_text(tree_classifier)
print(txt_tree)



plt.figure(figsize=(25,20))
plot_tree(tree_classifier,
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True)
plt.show()


# We can also estimate the probability that an instance belongs to a particular class.
print(tree_classifier.predict_proba([[4.5,2]]))

# CART Algorithm
# Classification and regression tree (CART) algorithm is used by Sckit-Learn to train decision trees.
# So what this algorithm does is firstly it splits the training set into two subsets using a single feature let’s say x and a threshold tx
# as in the earlier example our root node was “Petal Length”(x) and <= 2.45 cm(tx).
# Now you must be wondering how does it choose x and tx? It searches for a pair that will produce the purest subsets.
# Once the algorithm splits the training sets in two, it then splits the subsets with the same method and so on.
# This will stop when the max depth is reached (the hyperparameter which we set 2 earlier),
# or when it fails to find any other split that will reduce the impurity. There are a few other hyperparameters that control these stopping conditions