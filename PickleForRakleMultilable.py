import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
import time
import random  # Import random module

# Load the dataset
data = pd.read_csv('LungCancer32.csv')

X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Distribution of target labels
print("Distribution of target labels (DX-bone):")
print(y['DX-bone'].value_counts())
print("Distribution of target labels (DX-brain):")
print(y['DX-brain'].value_counts())
print("Distribution of target labels (DX-liver):")
print(y['DX-liver'].value_counts())

# Define RakEl algorithm with class weights
def rakel(X, y, k, m, strategy):
    n_labels = y.shape[1]
    ensemble = []
    labelsets = []

    if k > n_labels:
        raise ValueError("k must be smaller than or equal to the number of labels")

    if strategy not in ['disjoint', 'overlapping']:
        raise ValueError("strategy must be either 'disjoint' or 'overlapping'")

    fixed_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'  # Handle class imbalance
    }

    # Choose labelset indices based on strategy
    if strategy == 'disjoint':
        chunks = [list(range(n_labels))[i:i + k] for i in range(0, n_labels, k)]
        labelset_indices = random.choice(chunks)
    else:
        labelset_indices = random.sample(list(range(n_labels)), k)

    labelset = y.columns[labelset_indices]
    labelsets.append(labelset)

    clf = RandomForestClassifier(**fixed_params)
    clf.fit(X, y[labelset])
    ensemble.append(clf)

    return ensemble, labelsets

# Prediction function using RakEl ensemble
def predict_rakel(X, ensemble, labelsets):
    n_labels = len(labelsets[0])
    y_pred = np.zeros((X.shape[0], n_labels))

    for clf, labelset in zip(ensemble, labelsets):
        labelset_indices = y.columns.get_indexer(labelset)
        y_pred[:, labelset_indices] = clf.predict(X)

    return y_pred

# Save the model to a pickle file
def save_model(ensemble, labelsets, model_file='rakel_model.pkl'):
    with open(model_file, 'wb') as f:
        pickle.dump((ensemble, labelsets), f)
    print(f"Model saved to {model_file}")

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
        print(f"No model found at {model_file}. Training a new model.")
        return None, None

# Check if a trained model already exists, otherwise train a new one
model_file = 'rakel_model.pkl'
ensemble, labelsets = load_model(model_file)

if ensemble is None:
    # Train the model on the entire dataset
    start_train_time = time.time()
    ensemble, labelsets = rakel(X, y, k=3, m=1, strategy='disjoint')
    print("Training Time: {:.2f} seconds".format(time.time() - start_train_time))

    # Save the model after training
    save_model(ensemble, labelsets, model_file)

# Make predictions on the entire dataset using the loaded (or newly trained) model
start_test_time = time.time()
y_pred = predict_rakel(X, ensemble, labelsets)
print("Testing Time: {:.2f} seconds".format(time.time() - start_test_time))

# Convert predictions to binary (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Print the predictions and actual values
print("\nPredictions for the entire dataset:\n", y_pred_binary)
print("\nActual labels for the entire dataset:\n", y.values)

# Calculate accuracy, precision, recall, and F1-score for each label
accuracy_bone = accuracy_score(y['DX-bone'], y_pred_binary[:, 0])
accuracy_brain = accuracy_score(y['DX-brain'], y_pred_binary[:, 1])
accuracy_liver = accuracy_score(y['DX-liver'], y_pred_binary[:, 2])

precision_bone = precision_score(y['DX-bone'], y_pred_binary[:, 0])
recall_bone = recall_score(y['DX-bone'], y_pred_binary[:, 0])
f1_bone = f1_score(y['DX-bone'], y_pred_binary[:, 0])

precision_brain = precision_score(y['DX-brain'], y_pred_binary[:, 1])
recall_brain = recall_score(y['DX-brain'], y_pred_binary[:, 1])
f1_brain = f1_score(y['DX-brain'], y_pred_binary[:, 1])

precision_liver = precision_score(y['DX-liver'], y_pred_binary[:, 2])
recall_liver = recall_score(y['DX-liver'], y_pred_binary[:, 2])
f1_liver = f1_score(y['DX-liver'], y_pred_binary[:, 2])

# Print accuracy, precision, recall, and F1-score
print(f"\nAccuracy for DX-bone: {accuracy_bone:.2f}")
print(f"Precision for DX-bone: {precision_bone:.2f}")
print(f"Recall for DX-bone: {recall_bone:.2f}")
print(f"F1-Score for DX-bone: {f1_bone:.2f}\n")

print(f"Accuracy for DX-brain: {accuracy_brain:.2f}")
print(f"Precision for DX-brain: {precision_brain:.2f}")
print(f"Recall for DX-brain: {recall_brain:.2f}")
print(f"F1-Score for DX-brain: {f1_brain:.2f}\n")

print(f"Accuracy for DX-liver: {accuracy_liver:.2f}")
print(f"Precision for DX-liver: {precision_liver:.2f}")
print(f"Recall for DX-liver: {recall_liver:.2f}")
print(f"F1-Score for DX-liver: {f1_liver:.2f}")
