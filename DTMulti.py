# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 12:41:24 2023

@author: ftuha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:22:14 2023

@author: ftuha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score

start_time = time.time()

data = pd.read_csv('LungCancer32.csv')

#X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
#          'Histology', 'TNM', 'Reason_no_surgey']]
#y = data[['DX-bone', 'DX-brain', 'DX-liver']]


X = data[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

model = MultiOutputClassifier(DecisionTreeClassifier())  # Use DecisionTreeClassifier here

model_train_start_time = time.time()
model.fit(X_train, y_train)
model_train_end_time = time.time()

print("Training Time:", model_train_end_time - model_train_start_time)

y_pred_start_time = time.time()
y_pred = model.predict(X_test)
y_pred_end_time = time.time()

print("Testing Time:", y_pred_end_time - y_pred_start_time)

param_grid = {
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__max_features': ['auto', 'sqrt', 'log2'],

}

grid_start_time = time.time()

grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

grid_end_time = time.time()

print("Best Parameters:", grid_search.best_params_)

print("Grid Search Time:", grid_end_time - grid_start_time)
print("Training Time:", model_train_end_time - model_train_start_time)

metrics_start_time = time.time()

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

hamming = hamming_loss(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

metrics_end_time = time.time()

print("Test Time:", metrics_end_time - metrics_start_time)

print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")
print("Grid Search Time:", grid_end_time - grid_start_time)
print("Training Time:", model_train_end_time - model_train_start_time)
print("Testing Time:", y_pred_end_time - y_pred_start_time)