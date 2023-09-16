import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, hamming_loss, accuracy_score

import time

start_time = time.time()

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

# Import libraries
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, hamming_loss, accuracy_score
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV

# Define a base classifier (logistic regression)
base_clf = LogisticRegression()

# Define a parameter grid for the base classifier
param_grid = {
    'base_estimator__C': [0.01, 0.1, 1.0],
    'base_estimator__solver': ['lbfgs', 'liblinear', 'saga']
}

# Create an ensemble of 10 classifier chains with random orderings and grid search
chains = [GridSearchCV(ClassifierChain(base_clf, order='random',
                                       random_state=i), param_grid,
                       scoring='jaccard_samples', verbose=1) for i in range(10)]

# Fit each chain on the train set
train_start_time = time.time()
for chain in chains:
    chain.fit(X_train, y_train)
train_end_time = time.time()

# Predict on the test set using each chain
test_start_time = time.time()
Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
test_end_time = time.time()

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

# Calculate and print times
train_time = train_end_time - train_start_time
test_time = test_end_time - test_start_time
grid_search_time = train_time  # Total time taken for grid search is the same as train time
print(f"Train Time: {train_time:.3f} seconds")
print(f"Test Time: {test_time:.3f} seconds")
print(f"Grid Search Time: {grid_search_time:.3f} seconds")

total_time = time.time() - start_time
print(f"Total Execution Time: {total_time:.3f} seconds")
