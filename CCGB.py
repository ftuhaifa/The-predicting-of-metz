# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:48:47 2023

@author: ftuha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import time

# Load the dataset
data = pd.read_csv('LungCancer32.csv')

# Separate input features and target labels
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'TNM', 'Reason_no_surgey', 'Histology']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,
                                                    random_state=42)

base_gb = GradientBoostingClassifier(random_state=42)
chain = ClassifierChain(base_gb, order='random', random_state=42)

# Define hyperparameters and perform hyperparameter tuning
param_grid = {
    'base_estimator__n_estimators': [50, 100, 200],
    'base_estimator__max_depth': [3, 4, 5]
}
start_grid_search = time.time()
clf = GridSearchCV(chain, param_grid, cv=3)
clf.fit(X_train, y_train)
end_grid_search = time.time()

# Predict on the test set
start_test = time.time()
y_pred = clf.predict(X_test)
end_test = time.time()

# Print the best hyperparameters
print("Best Hyperparameters:", clf.best_params_)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Subset Accuracy:", accuracy_score(y_test, y_pred))

# Calculate timing information
train_time = end_grid_search - start_grid_search
test_time = end_test - start_test

print(f"Train Time: {train_time:.4f} seconds")
print(f"Grid Search Time: {train_time:.4f} seconds")
print(f"Test Time: {test_time:.4f} seconds")

from sklearn.metrics import f1_score, recall_score, precision_score

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
