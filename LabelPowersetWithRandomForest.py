# Import the function
from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv('LungCancer34.csv')

# Separate input features and target labels
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Normalize the data using Min-Max Normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y,
                                                    test_size=0.3,
                                                    random_state=42)



# Initialize LabelPowerset multi-label classifier with a RandomForest classifier
classifier = LabelPowerset(classifier=RandomForestClassifier(),
                           require_dense=[False,True])


param_grid = { 'classifier__n_estimators': [50, 100, 200],
              'classifier__max_depth': [None, 10, 20],
              'classifier__min_samples_split': [2, 5, 10] }

#Initialize GridSearchCV with the classifier and the parameter grid
#grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='f1_macro')
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
# Train
grid_search.fit(X_train,y_train)

# Predict
y_pred = grid_search.predict(X_test)

#Print the best parameters and score
print(grid_search.best_params_) 
print(grid_search.best_score_)

# Import the metrics


# Import the metrics
from sklearn.metrics import hamming_loss, f1_score, precision_recall_curve, accuracy_score

# Calculate Hamming Loss
hl = hamming_loss(y_test, y_pred)
print(f"Hamming Loss: {hl}")

# Calculate macro F1-Score, Precision and Recall
f1_macro = f1_score(y_test, y_pred, average="macro")
precision_macro = f1_score(y_test, y_pred, average="macro")
recall_macro = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1-Score: {f1_macro}")
print(f"Macro Precision: {precision_macro}")
print(f"Macro Recall: {recall_macro}")

# Calculate micro F1-Score, Precision and Recall
f1_micro = f1_score(y_test, y_pred, average="micro")
precision_micro = f1_score(y_test, y_pred, average="micro")
recall_micro = f1_score(y_test, y_pred, average="micro")
print(f"Micro F1-Score: {f1_micro}")
print(f"Micro Precision: {precision_micro}")
print(f"Micro Recall: {recall_micro}")

# Calculate average accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Average Accuracy: {acc}")

# Calculate Precision-Recall Curve for each label
for i in range(y.shape[1]):
    precision, recall, _ = precision_recall_curve(y[:, i], y_pred.toarray()[:, i])
    plt.plot(recall, precision, label=f"Label {i+1}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()





