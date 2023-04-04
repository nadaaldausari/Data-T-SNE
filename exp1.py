#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import glob


# In[2]:


# Load the data from CSV files
csv_files = glob.glob('data/*.csv')
dataframes = [pd.read_csv(file) for file in csv_files]
data = np.concatenate([df.values for df in dataframes], axis=0)


# In[3]:


# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)


# In[4]:


# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(data)


# In[5]:


# Visualize the results
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * 57  # one color per file
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    label = file.split('/')[-1].split('.')[0]  # extract the file name
    x = tsne_results[i*df.shape[0]:(i+1)*df.shape[0], 0]
    y = tsne_results[i*df.shape[0]:(i+1)*df.shape[0], 1]
    plt.scatter(x, y, c=colors[i], label=label, alpha=0.5)

plt.legend()
plt.show()


# In[ ]:


# Define a list to hold the t-SNE results for each CSV file
tsne_results_list = []

# Loop through the CSV files and compute t-SNE for each one
for file in csv_files:
    df = pd.read_csv(file)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df)
    tsne_results_list.append(tsne_results)

# Analyze the relationships
for i, tsne_i in enumerate(tsne_results_list):
    for j, tsne_j in enumerate(tsne_results_list):
        if i == j:
            continue
        dist = np.linalg.norm(tsne_i - tsne_j)
        print(f"Distance between {csv_files[i]} and {csv_files[j]}: {dist}")


# In[ ]:




