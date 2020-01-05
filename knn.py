# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 20:14:08 2020

@author: M.Dhia
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


location = r'C:\Users\M.Dhia\Desktop\Finale\EtudiantDataBaseFinale.csv'

dataset = pd.read_csv(location)


X = dataset.iloc[:,1:21].values
norm_data = MinMaxScaler() #initialisation
X = norm_data.fit_transform(X)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances,axis=0)

distances = distances[:,1]

plt.plot(distances)

