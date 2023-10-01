# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:06:41 2023

@author: ftuha
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('LungCancer32.csv')

# Separate input features and target labels
X = data[['Age', 'Sex', 'Behavior', 'Primary_Site', 'Laterality', 'Race',
          'Histology', 'TNM', 'Reason_no_surgey']]
y = data[['DX-bone', 'DX-brain', 'DX-liver']]

# Calculate correlations between labels
label_correlations = y.corr()

# Create a graph from the correlations
G = nx.Graph()

# Add nodes for each label
for label in label_correlations.columns:
    G.add_node(label)

# Add edges for correlated labels
for i in range(len(label_correlations.columns)):
    for j in range(i + 1, len(label_correlations.columns)):
        correlation = label_correlations.iloc[i, j]
        if abs(correlation) >= 0.1:  # Adjust the correlation threshold as needed
            G.add_edge(label_correlations.columns[i], label_correlations.columns[j], weight=correlation)

# Visualize the graph
pos = nx.spring_layout(G, seed=42)  # You can use different layout algorithms
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_color='black')

# Add edge labels with correlation values
edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Label Correlation Graph")
plt.show()
