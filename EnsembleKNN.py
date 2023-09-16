# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:55:59 2023

@author: ftuha
"""


from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, hamming_loss, accuracy_score


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer




# Load the dataset
data = pd.read_csv('LungCancer32.csv')

# Define features and target variables
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=42)

# Set up parameter grid for grid search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

# Initialize k-NN classifier
knn = KNeighborsClassifier()

# Create GridSearchCV instance
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3,
                           scoring='accuracy')

# Initialize an empty DataFrame for ensemble predictions
ensemble_pred = pd.DataFrame(index=y_test.index, columns=y.columns)

# Fit GridSearchCV on training data for each label
for label in y.columns:
    grid_search.fit(X_train, y_train[label])
    print(f"Best parameters for {label}: {grid_search.best_params_}")

    # Use the best estimator for predictions
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)

    accuracy = accuracy_score(y_test[label], y_pred)
    hamming = hamming_loss(y_test[label], y_pred)

    print(f'Accuracy for {label}: {accuracy:.2f}')
    print(f'Hamming Loss for {label}: {hamming:.2f}')
    print('---')

    # Store the predictions in the ensemble_pred DataFrame
    ensemble_pred[label] = y_pred

# Evaluate the ensemble's performance
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_hamming = hamming_loss(y_test, ensemble_pred)

print(f'Ensemble Accuracy: {ensemble_accuracy:.2f}')
print(f'Ensemble Hamming Loss: {ensemble_hamming:.2f}')


# Calculate various metrics using y_pred
f1_macro = f1_score(y_test, ensemble_pred, average='macro')
f1_micro = f1_score(y_test, ensemble_pred, average='micro')

recall_macro = recall_score(y_test, ensemble_pred, average='macro')
recall_micro = recall_score(y_test, ensemble_pred, average='micro')

precision_macro = precision_score(y_test, ensemble_pred, average='macro')
precision_micro = precision_score(y_test, ensemble_pred, average='micro')

hamming = hamming_loss(y_test, ensemble_pred)

print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {hamming}")
