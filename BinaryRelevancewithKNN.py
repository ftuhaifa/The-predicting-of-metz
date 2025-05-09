# -*- coding: utf-8 -*-
"""

Fatimah Altuhaifa: Binary Relevance with KNN
"""

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Multi-label classification method
from skmultilearn.problem_transform import BinaryRelevance

# Classifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import (
    f1_score, recall_score, precision_score, hamming_loss,
    accuracy_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('LungCancer32.csv')

# Select input features and target labels
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Normalize input features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.3, random_state=42
)

# Define parameter grid for KNN
param_grid = {
    'classifier__n_neighbors': [3, 5, 7],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

# Wrap KNN in Binary Relevance
br_classifier = BinaryRelevance(classifier=KNeighborsClassifier(), require_dense=[True, True])

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=br_classifier,
                           param_grid=param_grid,
                           scoring='f1_macro',
                           cv=3)

# Time training duration
start_time = time.time()

# Train the model with grid search
grid_search.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

# Get best model from grid search
best_classifier = grid_search.best_estimator_

# Output best parameters
print("Best Parameters:", grid_search.best_params_)

# Make predictions on test data
y_pred = best_classifier.predict(X_test)

# Convert to arrays if sparse
if hasattr(y_pred, "toarray"):
    y_pred = y_pred.toarray()
if hasattr(y_test, "toarray"):
    y_test = y_test.toarray()

# Evaluation Metrics
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')
hamming = hamming_loss(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# Print metrics
print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")
print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")
print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")
print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")
print(f"Training Time: {training_time:.2f} seconds")

# Plot Precision-Recall curves
for i in range(y.shape[1]):
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
    plt.plot(recall, precision, label=f"Label {i+1}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Each Label")
plt.legend()
plt.grid(True)
plt.show()


# Time training duration (Grid Search)
import time
# Time training duration (Grid Search)
start_time_grid_search = time.time()
grid_search.fit(X_train, y_train)
end_time_grid_search = time.time()
grid_search_time = end_time_grid_search - start_time_grid_search

# Get best model from grid search
best_classifier = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Time testing duration
start_time_testing = time.time()

# Make predictions on test data
y_pred = best_classifier.predict(X_test)

# Convert to dense arrays if needed
if hasattr(y_pred, "toarray"):
    y_pred = y_pred.toarray()
if hasattr(y_test, "toarray"):
    y_test = y_test.toarray()

# Calculate performance metrics
hl = hamming_loss(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')
acc = accuracy_score(y_test, y_pred)

# End time for testing
end_time_testing = time.time()
test_time = end_time_testing - start_time_testing

# Print performance metrics
print(f"Hamming Loss: {hl}")
print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")
print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")
print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")
print(f"Average Accuracy: {acc}")

# Print computation times
print(f"Grid Search Time: {grid_search_time:.4f} seconds")
print(f"Test Time: {test_time:.4f} seconds")
