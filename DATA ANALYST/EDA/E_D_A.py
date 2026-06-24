# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:48:55 2026

@author: hp
"""

# Iris Flower Classification with EDA


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 2. Exploratory Data Analysis (EDA)
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nClass Distribution:")
print(df['species'].value_counts())

# Pairplot visualization
sns.pairplot(df, hue="species", diag_kind="kde")
plt.show()

# Boxplot example
sns.boxplot(x="species", y="sepal length (cm)", data=df)
plt.show()

# Scatterplot example
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="species", data=df)
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# 3. Train/Test Split
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Model (Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
