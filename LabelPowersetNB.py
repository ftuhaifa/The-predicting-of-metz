# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:53:04 2023

@author: ftuha
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import hamming_loss, f1_score, precision_recall_curve, accuracy_score

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

# Initialize LabelPowerset multi-label classifier with Gaussian Naive Bayes
classifier = LabelPowerset(classifier=GaussianNB())

param_grid = {
    'classifier__priors': [None, [0.3, 0.4, 0.3], [0.1, 0.8, 0.1]]  # Example priors, you can customize
}

# Initialize GridSearchCV with the classifier and the parameter grid
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
# Train
grid_search.fit(X_train, y_train)

# Predict
y_pred = grid_search.predict(X_test)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)

# Calculate and print metrics
hl = hamming_loss(y_test, y_pred)
print("Hamming Loss:", hl)

# Rest of the metrics calculations and plotting code...
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

hamming = hamming_loss(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")
