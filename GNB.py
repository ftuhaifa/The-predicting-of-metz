# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:01:13 2023

@author: ftuha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score
import time


data = pd.read_csv('LungCancer32.csv')

X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MultiOutputClassifier(GaussianNB())  # Gaussian Naive Bayes classifier
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

param_grid = {
    'estimator__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],  # Smoothing parameter
    # Add other Gaussian Naive Bayes specific hyperparameters as needed
}

start_grid_search_time = time.time()

grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)

end_grid_search_time = time.time()
grid_search_time = end_grid_search_time - start_grid_search_time

print("grid search Time: {:.2f} seconds".format(grid_search_time))



# Measure the time taken for training
start_train_time = time.time()
grid_search.fit(X_train_scaled, y_train)



end_train_time = time.time()
training_time = end_train_time - start_train_time
print("Training Time: {:.2f} seconds".format(training_time))


best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)




# Predict labels for the test set using the best estimator
start_test_time = time.time()
y_pred = best_model.predict(X_test_scaled)

end_test_time = time.time()
testing_time = end_test_time - start_test_time
print("Testing Time: {:.2f} seconds".format(testing_time))




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
