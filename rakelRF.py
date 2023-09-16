import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import random



import time

start_time = time.time()

# Load the dataset (replace 'LungCancer32.csv' with your dataset filename)
data = pd.read_csv('LungCancer32.csv')

X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Define the RakEl algorithm with grid search
def rakel(X, y, k, m, strategy):
    n_labels = y.shape[1]
    ensemble = []
    labelsets = []

    if k > n_labels:
        raise ValueError("k must be smaller than or equal to the number of labels")

    if strategy not in ['disjoint', 'overlapping']:
        raise ValueError("strategy must be either 'disjoint' or 'overlapping'")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    for i in range(m):
        if strategy == 'disjoint':
            chunks = [list(range(n_labels))[i:i + k] for i in range(0, n_labels, k)]
            labelset_indices = random.choice(chunks)
        else:
            labelset_indices = random.sample(list(range(n_labels)), k)

        labelset = y.columns[labelset_indices]  # Convert indices to column names
        labelsets.append(labelset)
    
    
        start_grid_search_time = time.time()
        clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
        
  
        end_grid_search_time = time.time()
        grid_search_time = end_grid_search_time - start_grid_search_time

        print("grid search Time: {:.2f} seconds".format(grid_search_time))
        
        clf.fit(X, y[labelset])  # Use labelset column names for indexing
        ensemble.append(clf)

    return ensemble, labelsets

# Define a function to make predictions using the RakEl algorithm
def predict_rakel(X, ensemble, labelsets):
    n_labels = len(labelsets[0])  # Number of labels in each labelset
    y_pred = np.zeros((X.shape[0], n_labels))

    for clf, labelset in zip(ensemble, labelsets):
        labelset_indices = y.columns.get_indexer(labelset)  # Get indices of label columns
        y_pred[:, labelset_indices] = clf.predict(X)

    return y_pred

# Split the data into train and test sets with a ratio of 0.8/0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Apply the RakEl algorithm with k=3, m=10 and disjoint strategy
# Measure the time taken for training
start_train_time = time.time()
ensemble, labelsets = rakel(X_train, y_train, k=3, m=10, strategy='disjoint')

end_train_time = time.time()
training_time = end_train_time - start_train_time
print("Training Time: {:.2f} seconds".format(training_time))



# Predict labels for the test set using the best estimator
start_test_time = time.time()
# Make predictions on the test set using the RakEl algorithm
y_pred = predict_rakel(X_test, ensemble, labelsets)

end_test_time = time.time()
testing_time = end_test_time - start_test_time
print("Testing Time: {:.2f} seconds".format(testing_time))

# Evaluate the performance using hamming loss metric
loss = hamming_loss(y_test, y_pred)
print(f"The hamming loss is {loss:.4f}")

# Additional evaluation metrics
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

acc = accuracy_score(y_test, y_pred)

print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {loss}")
print(f"Average Accuracy: {acc}")

# Print the best hyperparameters for each classifier
for i, clf in enumerate(ensemble):
    print(f"Best Hyperparameters for Classifier {i+1}: {clf.best_params_}")
