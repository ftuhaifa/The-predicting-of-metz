# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 21:19:10 2023

@author: ftuha
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, recall_score, precision_score
import time

# Load data

# Load the dataset
data = pd.read_csv('LungCancer32.csv')

# Define features and target variables
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Create pipeline with DecisionTreeClassifier
pipeline = Pipeline([
    ('feature_selection', SelectKBest(chi2, k='all')),  # Set k to 'all' to use all features
    ('classification', OneVsRestClassifier(DecisionTreeClassifier()))  # Use DecisionTreeClassifier here
])

# Define hyperparameters for grid search
param_grid = {
    'classification__estimator__max_depth': [None, 10, 20, 30],
    'classification__estimator__min_samples_split': [2, 5, 10],
}

# Create GridSearchCV object with the pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# Measure training time
start_time = time.time()
grid_search.fit(X_train, y_train)
train_time = time.time() - start_time

# Measure testing time
start_time = time.time()
y_pred = grid_search.predict(X_test)
test_time = time.time() - start_time

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
hamming_loss_value = hamming_loss(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy}")
print(f"Hamming Loss: {hamming_loss_value}")
print(f"F1 (Macro): {f1_macro}")
print(f"F1 (Micro): {f1_micro}")
print(f"Recall (Macro): {recall_macro}")
print(f"Recall (Micro): {recall_micro}")
print(f"Precision (Macro): {precision_macro}")
print(f"Precision (Micro): {precision_micro}")
print(f"Training Time: {train_time} seconds")
print(f"Testing Time: {test_time} seconds")
print(f"Grid Search Time: {grid_search.cv_results_['mean_fit_time'].sum()} seconds")
