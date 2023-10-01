import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score

# Record the start time for data loading
start_time = time.time()

# Load the data
data = pd.read_csv('LungCancer32.csv')

# Calculate data loading time
data_loading_time = time.time() - start_time

# Separate features and target variables
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Initialize a logistic regression classifier
logistic_classifier = LogisticRegression()

# Wrap it with the OneVsRestClassifier
ovr_classifier = OneVsRestClassifier(logistic_classifier)

# Record the start time for training
start_time = time.time()
# Fit the one-vs-rest classifier on the training data
ovr_classifier.fit(X_train, y_train)
training_time = time.time() - start_time

# Record the start time for predicting
start_time = time.time()
# Predict probabilities for each class on the test data
y_pred_prob = ovr_classifier.predict_proba(X_test)
testing_time = time.time() - start_time

# Define hyperparameter grid for grid search
param_grid = {
    'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'estimator__penalty': ['l1', 'l2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(ovr_classifier, param_grid, cv=5)

# Record the start time for grid search
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time

best_model = grid_search.best_estimator_

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Predict using the best model
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

hamming = hamming_loss(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

# Print evaluation metrics
print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")

# Print the timings
print(f"Data Loading Time: {data_loading_time:.2f} seconds")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Testing Time: {testing_time:.2f} seconds")
print(f"Grid Search Time: {grid_search_time:.2f} seconds")
