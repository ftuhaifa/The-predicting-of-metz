# -*- coding: utf-8 -*-
"""
Created on Sat May 24 19:25:30 2025

@author: Fatimah Altuhaifa
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from captum.attr import Saliency, NoiseTunnel

# Load data
data_train = pd.read_csv('LungCancer32.csv')
data_verification = pd.read_csv('LungCancer06.csv')

# Filter relevant columns
data_train = data_train[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM', 'DX-bone', 'DX-brain', 'DX-liver']]
data_verification = data_verification[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM', 'DX-bone', 'DX-brain', 'DX-liver']]

# Convert to numeric
data_train = data_train.apply(pd.to_numeric, errors='coerce')
data_verification = data_verification.apply(pd.to_numeric, errors='coerce')

# Features and labels
X = data_train[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM']]
y = data_train[['DX-bone', 'DX-brain', 'DX-liver']]

X.fillna(X.mean(), inplace=True)
y.fillna(y.mode().iloc[0], inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
k_range = range(2, 10)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data_train['Cluster'] = clusters

# SMOTE per cluster
cluster_counts = data_train['Cluster'].value_counts()
for cluster in range(optimal_k):
    cluster_data = data_train[data_train['Cluster'] == cluster]
    if cluster_counts[cluster] < 2000:
        X_cluster = cluster_data[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM']]
        y_cluster = cluster_data[['DX-bone', 'DX-brain', 'DX-liver']]
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_cluster, y_cluster)
        smote_df = pd.DataFrame(X_resampled, columns=X_cluster.columns)
        smote_df['Cluster'] = cluster
        y_df = pd.DataFrame(y_resampled, columns=y_cluster.columns)
        data_train = pd.concat([data_train, smote_df], ignore_index=True)
        data_train.loc[data_train.index[-len(y_df):], y_df.columns] = y_df.values

# Define BNN
class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

# Train models per cluster
cluster_models = {}
labels = ['DX-bone', 'DX-brain', 'DX-liver']
feature_names = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM']

from captum.attr import Saliency, NoiseTunnel

for cluster in range(optimal_k):
    print(f"\nTraining Model for Cluster {cluster}...")
    cluster_data = data_train[data_train['Cluster'] == cluster]
    X_cluster = scaler.transform(cluster_data[feature_names])
    y_cluster = cluster_data[labels]

    X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    model = BayesianNN(X_train.shape[1], y_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    print(f"\nRunning SmoothGrad for Cluster {cluster}...\n")



    

    model.eval()
    saliency = Saliency(model)
    smooth_grad = NoiseTunnel(saliency)

    X_test_np = X_test_tensor.numpy()
    y_test_np = y_test_tensor.numpy()

    multilabel_patterns = [''.join(map(str, map(int, row))) for row in y_test_np]
    unique_patterns = sorted(set(multilabel_patterns))

    for pattern in unique_patterns:
        indices = [i for i, p in enumerate(multilabel_patterns) if p == pattern]
        if len(indices) < 2:
            continue

        X_subset = torch.tensor(X_test_np[indices], dtype=torch.float32)
        all_attr = []
        for i in range(len(labels)):
            attr = smooth_grad.attribute(
                X_subset,
                nt_type='smoothgrad',
                stdevs=0.1,
                nt_samples=20,
                target=i
            )
            all_attr.append(attr.detach().numpy())

        all_attr = np.stack(all_attr, axis=0)
        joint_attr = np.mean(np.abs(all_attr), axis=(0, 1))
        
        #\U0001F4CA

        print(f"\n SmoothGrad for Multilabel Class {pattern} (Cluster {cluster}):")
        for name, val in zip(feature_names, joint_attr):
            print(f"{name}: {val:.6f}")

        plt.figure(figsize=(8, 5))
        bars = plt.barh(feature_names, joint_attr)
        plt.xlabel("Mean Absolute Attribution (SmoothGrad)")
        plt.title(f"CBNN - SmoothGrad for Multilabel Class {pattern} (Cluster {cluster})")
        plt.gca().invert_yaxis()
        for bar, val in zip(bars, joint_attr):
            plt.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va='center')
        plt.tight_layout()
        plt.show()
