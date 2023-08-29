import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

from sklearn.ensemble import GradientBoostingClassifier



# Load the dataset
data = pd.read_csv('LungCancer34.csv')

# Separate input features and target labels
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'TNM', 'Reason_no_surgey', 'Histology']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,
                                                    random_state=42)

base_gb = GradientBoostingClassifier(random_state=42)
chain = ClassifierChain(base_gb, order='random', random_state=42)

# Define hyperparameters and perform hyperparameter tuning
param_grid = {'base_estimator__n_estimators': [10, 50, 100],
              'base_estimator__learning_rate': [0.01, 0.1, 0.5]}
clf = GridSearchCV(chain, param_grid, cv=5)
clf.fit(X_train, y_train)


# Predict on the test set
y_pred = clf.predict(X_test)

# Print the best hyperparameters
print("Best Hyperparameters:", clf.best_params_)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Subset Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import f1_score
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

from sklearn.metrics import recall_score
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

from sklearn.metrics import precision_score
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

hamming = hamming_loss(y_test, y_pred)

# Calculate average accuracy
acc = accuracy_score(y_test, y_pred)

print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")
print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")
print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")
print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")
