# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:54:52 2023
@author: ftuha
"""

# import the package
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import numpy as np

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report, hamming_loss, precision_recall_curve,  accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('LungCancer32.csv')

# Separate input features and target labels
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Normalize the data using Min-Max Normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Define the parameter grid for SVM classifier
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

# Initialize Binary Relevance multi-label classifier with SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
classifier = BinaryRelevance(SVC(random_state=42, probability=True))

# Initialize GridSearchCV with 10-fold cross-validation and F1-score scoring
grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring='f1_macro')

# Train the classifier on the training data
grid_search.fit(X_train, y_train)

# Get the best classifier from the grid search
best_classifier = grid_search.best_estimator_

# Print the best parameters from the grid search
print("Best Parameters:", grid_search.best_params_)

# Predict the labels on the test data using the best classifier
y_pred = best_classifier.predict(X_test)

# Calculate F1-score
from sklearn.metrics import f1_score
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

# Calculate Recall
from sklearn.metrics import recall_score
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

# Calculate Precision
from sklearn.metrics import precision_score
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

# Calculate Hamming Loss
hamming = hamming_loss(y_test, y_pred)

# Calculate Average Accuracy
acc = accuracy_score(y_test, y_pred)

print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")

# Calculate Precision-Recall Curve for each label
for i in range(y.shape[1]):
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
    plt.plot(recall, precision, label=f"Label {i+1}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
