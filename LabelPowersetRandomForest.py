# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:54:02 2023

@author: ftuha
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score, recall_score, precision_score, accuracy_score
import time

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

# Initialize LabelPowerset multi-label classifier with Random Forest
classifier = LabelPowerset(classifier=RandomForestClassifier())

param_grid = {
    'classifier__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'classifier__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Initialize GridSearchCV with the classifier and the parameter grid
grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring='accuracy')

# Measure start time for grid search
start_time_grid_search = time.time()

# Train
# Measure the time taken for training
start_train_time = time.time()
grid_search.fit(X_train, y_train)

end_train_time = time.time()
training_time = end_train_time - start_train_time
print("Training Time: {:.2f} seconds".format(training_time))

# Measure end time for grid search
end_time_grid_search = time.time()

# Calculate grid search time
grid_search_time = end_time_grid_search - start_time_grid_search

# Predict
y_pred = grid_search.predict(X_test)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)

# Measure start time for testing
start_time_testing = time.time()

# Calculate metrics
hl = hamming_loss(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')
hamming = hamming_loss(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# Measure end time for testing
end_time_testing = time.time()

# Calculate testing time
test_time = end_time_testing - start_time_testing

# Print additional metrics and times
print(f"Hamming Loss: {hl}")
print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")
print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")
print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")
print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")

# Print times
print(f"Grid Search Time: {grid_search_time:.4f} seconds")
print(f"Test Time: {test_time:.4f} seconds")
