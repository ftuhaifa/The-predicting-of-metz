# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:22:31 2023
@author: ftuha
"""

# Import libraries
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, hamming_loss, accuracy_score
from sklearn.naive_bayes import GaussianNB

# Load the data
data = pd.read_csv('LungCancer32.csv')

# Separate features and target variables
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Define a base classifier (Gaussian Naive Bayes)
base_clf = GaussianNB()

# Create an ensemble of 10 classifier chains with random orderings
chains = [ClassifierChain(base_clf, order='random', random_state=i) for i in range(10)]

# Measure the training time
start_time = time.time()
# Fit each chain on the train set
for chain in chains:
    chain.fit(X_train, y_train)
training_time = time.time() - start_time

# Predict on the test set using each chain
start_time = time.time()
Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
testing_time = time.time() - start_time

# Calculate the Jaccard score for each chain
chain_scores = [jaccard_score(y_test, Y_pred_chain, average='samples') for Y_pred_chain in Y_pred_chains]

# Print the scores of each chain
print('Scores of individual chains:')
for i, score in enumerate(chain_scores):
    print(f'Chain {i+1}: {score:.3f}')

# Create an ensemble prediction by averaging the binary predictions of the chains
Y_pred_ensemble = Y_pred_chains.mean(axis=0)

# Apply a threshold of 0.5 to convert the probabilities to labels
Y_pred_ensemble = (Y_pred_ensemble >= 0.5).astype(int)

# Calculate the Jaccard score for the ensemble
ensemble_score = jaccard_score(y_test, Y_pred_ensemble, average='samples')

# Print the score of the ensemble
print('Score of ensemble:', ensemble_score)

# Calculate evaluation metrics
f1_macro = f1_score(y_test, Y_pred_ensemble, average='macro')
f1_micro = f1_score(y_test, Y_pred_ensemble, average='micro')

recall_macro = recall_score(y_test, Y_pred_ensemble, average='macro')
recall_micro = recall_score(y_test, Y_pred_ensemble, average='micro')

precision_macro = precision_score(y_test, Y_pred_ensemble, average='macro')
precision_micro = precision_score(y_test, Y_pred_ensemble, average='micro')

hamming = hamming_loss(y_test, Y_pred_ensemble)

acc = accuracy_score(y_test, Y_pred_ensemble)

# Print evaluation metrics
print(f"Macro Precision: {precision_macro:.3f}")
print(f"Micro Precision: {precision_micro:.3f}")

print(f"Macro Recall: {recall_macro:.3f}")
print(f"Micro Recall: {recall_micro:.3f}")

print(f"Macro F1-Score: {f1_macro:.3f}")
print(f"Micro F1-Score: {f1_micro:.3f}")

print(f"Hamming Loss: {hamming:.3f}")
print(f"Average Accuracy: {acc:.3f}")

# Print timing information
print(f"Training Time: {training_time:.2f} seconds")
print(f"Testing Time: {testing_time:.2f} seconds")


