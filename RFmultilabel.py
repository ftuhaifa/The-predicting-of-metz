# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:22:14 2023

@author: ftuha
"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('LungCancer32.csv')

X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

model = MultiOutputClassifier(RandomForestClassifier())  # You can replace with other classifiers
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [None, 10, 20],
    # Add other hyperparameters as needed
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)



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
