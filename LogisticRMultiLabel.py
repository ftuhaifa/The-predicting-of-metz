# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:11:42 2023

@author: ftuha
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the data
data = pd.read_csv('LungCancer32.csv')

# Separate features and target variables
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Initialize a MultiOutputClassifier with LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
model = MultiOutputClassifier(LogisticRegression())

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the target variables
y_pred = model.predict(X_test)

# Define a grid of hyperparameters for GridSearchCV
param_grid = {
    'estimator__C': [0.1, 1, 10],  # Regularization parameter
    'estimator__solver': ['lbfgs', 'liblinear']  # Solver algorithm
    # Add other hyperparameters as needed
}

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Calculate evaluation metrics
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

hamming = hamming_loss(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

# Print evaluation metrics
print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")
