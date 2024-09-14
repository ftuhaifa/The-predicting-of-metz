# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 03:04:45 2024

@author: ftuha
"""

import pickle
import pandas as pd
import numpy as np
import os

# Load the model from a pickle file
def load_model(model_file='rakel_model.pkl'):
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                ensemble, labelsets = loaded_data
                print(f"Model loaded from {model_file}")
                return ensemble, labelsets
            else:
                raise ValueError("Loaded data does not have the expected structure.")
    else:
        print(f"No model found at {model_file}. Please ensure the file exists.")
        return None, None

# Prediction function using RakEl ensemble
def predict_rakel(X, ensemble, labelsets, y_columns):
    n_labels = len(labelsets[0])
    y_pred = np.zeros((X.shape[0], n_labels))

    for clf, labelset in zip(ensemble, labelsets):
        labelset_indices = y_columns.get_indexer(labelset)  # Get indices of label columns
        y_pred[:, labelset_indices] = clf.predict(X)

    return y_pred

# Assuming 'rakel_model.pkl' exists in the current directory
model_file = 'rakel_model.pkl'
ensemble, labelsets = load_model(model_file)

# Load the new data to make predictions (replace 'NewLungCancerData.csv' with your filename)
data = pd.read_csv('LungCancer32.csv')

# Assuming the same features as used in training
X_new = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race', 'Histology', 'TNM', 'Reason_no_surgey']]
y_columns = pd.Index(['DX-bone', 'DX-brain', 'DX-liver'])  # The label columns used in training

if ensemble is not None:
    # Make predictions using the loaded model
    y_pred = predict_rakel(X_new, ensemble, labelsets, y_columns)
    
    # Print predictions
    print("\nPredictions for the new dataset:\n", y_pred)
