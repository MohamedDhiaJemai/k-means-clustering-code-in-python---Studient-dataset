# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:51:51 2020

@author: M.Dhia
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

location = r'C:\Users\M.Dhia\Desktop\Finale\EtudiantDataBaseFinale.csv'

dataset = pd.read_csv(location)
# list(data) or 
#list(dataset.columns) 
#dataset.info()
#descriptive statistics of the dataset
#dataset.describe().transpose()


X = dataset.iloc[:,1:21].values



# Using the elbow method to find the optimal number of clusters
#wcss = []
#for i in range(1, 40):
#    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)
#plt.plot(range(1, 40), wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()

# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 20, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 15, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 15, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 15, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 15, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 15, c = 'magenta', label = 'Cluster 5')

plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 15, c = 'Black', label = 'Cluster 6')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 15, c = 'Aqua', label = 'Cluster 7')
plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 15, c = 'Orange', label = 'Cluster 8')
plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s = 15, c = 'Azure', label = 'Cluster 9')
plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9, 1], s = 15, c = 'Beige', label = 'Cluster 10')

plt.scatter(X[y_kmeans == 10, 0], X[y_kmeans == 10, 1], s = 15, c = 'red', label = 'Cluster 11')
plt.scatter(X[y_kmeans == 11, 0], X[y_kmeans == 11, 1], s = 15, c = 'blue', label = 'Cluster 12')
plt.scatter(X[y_kmeans == 12, 0], X[y_kmeans == 12, 1], s = 15, c = 'green', label = 'Cluster 13')
plt.scatter(X[y_kmeans == 13, 0], X[y_kmeans == 13, 1], s = 15, c = 'cyan', label = 'Cluster 14')
plt.scatter(X[y_kmeans == 14, 0], X[y_kmeans == 14, 1], s = 15, c = 'magenta', label = 'Cluster 15')

plt.scatter(X[y_kmeans == 15, 0], X[y_kmeans == 15, 1], s = 15, c = 'red', label = 'Cluster 16')
plt.scatter(X[y_kmeans == 16, 0], X[y_kmeans == 16, 1], s = 15, c = 'blue', label = 'Cluster 17')
plt.scatter(X[y_kmeans == 17, 0], X[y_kmeans == 17, 1], s = 15, c = 'green', label = 'Cluster 18')
plt.scatter(X[y_kmeans == 18, 0], X[y_kmeans == 18, 1], s = 15, c = 'cyan', label = 'Cluster 19')
plt.scatter(X[y_kmeans == 19, 0], X[y_kmeans == 19, 1], s = 15, c = 'magenta', label = 'Cluster 20')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'yellow', label = 'Centroid')



plt.title('Clusters of Studients')
plt.xlabel('Number of Cluster')
plt.ylabel('ENote')
plt.legend()
plt.show()




######### Affichage de nombre de cluster

cluster_labels = np.unique(y_kmeans)
print (cluster_labels)
######## 

