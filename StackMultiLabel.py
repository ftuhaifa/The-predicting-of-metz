import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score


# Import libraries
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier



data = pd.read_csv('LungCancer32.csv')

X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base classifiers
#base_classifiers = [
#    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#    ('gnb', GaussianNB()),
#    ('lr', LogisticRegression(max_iter=1000, random_state=42))
#]

# Initialize the stacking classifier with the base classifiers
#model = StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())


# Define base classifiers for stacking
estimators_list = [
    ('ExtraTrees', ExtraTreesClassifier(n_estimators=30, class_weight="balanced", random_state=42)),
    ('linearSVC', LinearSVC(class_weight='balanced'))
]

# Define meta-learner for stacking
final_estimator = LogisticRegression(solver='lbfgs', max_iter=300)

# Define stacking classifier with OVR strategy
stacking_clf = OneVsRestClassifier(StackingClassifier(estimators=estimators_list,
                                                      final_estimator=final_estimator))

# Fit the model on the data
stacking_clf.fit(X, y)

# Predict on new data
y_pred = stacking_clf.predict(X_test)


f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')

hamming = hamming_loss(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

print(f"Macro Precision: {precision_macro}")
print(f"Micro Precision: {precision_micro}")

print(f"Macro Recall: {recall_macro}")
print(f"Micro Recall: {recall_micro}")

print(f"Macro F1-Score: {f1_macro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Hamming Loss: {hamming}")
print(f"Average Accuracy: {acc}")


