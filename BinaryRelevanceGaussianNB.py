# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:54:52 2023

@author: ftuha
"""

# import the package
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression
import numpy as np

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report, hamming_loss, precision_recall_curve,  accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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


# Define the parameter grid for var_smoothing
#param_grid = {'classifier__var_smoothing': np.logspace(0,-9, num=100)}

# initialize Binary Relevance multi-label classifier
from sklearn.naive_bayes import GaussianNB
# with logistic regression base classifier
classifier = BinaryRelevance(GaussianNB())

# Initialize GridSearchCV with 10-fold cross-validation and accuracy scoring

#grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')

# train the classifier on the training data
classifier.fit(X_train, y_train)

# predict the labels on the test data
y_pred = classifier.predict(X_test)

#Print the best parameters and score
#print(grid_search.best_params_) 
#print(grid_search.best_score_)

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

# Calculate average accuracy
print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")

# Calculate Precision-Recall Curve for each label
for i in range(y.shape[1]):
    precision, recall, _ = precision_recall_curve(y[:, i], y_pred.toarray()[:, i])
    plt.plot(recall, precision, label=f"Label {i+1}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


