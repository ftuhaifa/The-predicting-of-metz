# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:05:48 2023

@author: ftuha
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:43:02 2023

@author: ftuha
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_multilabel_classification
import time

start_time = time.time()

# Load your dataset
data = pd.read_csv('LungCancer32.csv')
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Define the model with hyperparameters as function parameters
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Create a KerasClassifier for use in GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters and their possible values for tuning
param_grid = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'epochs': [50, 100, 150],
    'batch_size': [16, 32, 64]
}

# Use GridSearchCV for hyperparameter tuning
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy',
                    cv=5)
grid_result = grid.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters: ", grid_result.best_params_)

# Get the best model with the best hyperparameters
best_model = grid_result.best_estimator_.model

print(grid_result.best_estimator_.model)
print("**************************************")

# Fit the best model
best_model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'],
               batch_size=grid_result.best_params_['batch_size'], verbose=0)

# Make predictions on the test set
yhat = best_model.predict(X_test)
yhat = yhat.round()

# Calculate subset accuracy
subset_acc = accuracy_score(y_test, yhat)
print("Subset Accuracy: ", subset_acc)

# Calculate hamming loss
hl = hamming_loss(y_test, yhat)
print("Hamming Loss: ", hl)

# Generate classification report
class_report = classification_report(y_test, yhat)
print("Classification Report:\n", class_report)

# Evaluate the model
