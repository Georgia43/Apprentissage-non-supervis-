"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = '../artificial/'
name="xclara.arff"
#name="engytime.arff"




#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Variables for silhouette scores
best_k = None
best_score = -1
scores = []


inertie_list = []
k_list = range(2,25)
for k in k_list:

    
    tps1 = time.time()
    #k=3
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    inertie_list.append(inertie)
    centroids = model.cluster_centers_

silhouette_scores = []
# davies_bouldin_indices = []
# calinski_harabasz_indices = []
k_list = range(2, 10)  

for k in k_list:
    print("------------------------------------------------------")
    tps1 = time.time()
    
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    labels = model.fit_predict(datanp)

    tps2 = time.time()

    silhouette_score = metrics.silhouette_score(datanp, labels)
    # davies_bouldin_index = metrics.davies_bouldin_score(datanp, labels)
    # calinski_harabasz_index = metrics.calinski_harabasz_score(datanp, labels)

    silhouette_scores.append(silhouette_score)
    # davies_bouldin_indices.append(davies_bouldin_index)
    # calinski_harabasz_indices.append(calinski_harabasz_index)

    print(f"Nb clusters = {k}, runtime = {round((tps2 - tps1) * 1000, 2)} ms")
    print(f"Silhouette Score: {silhouette_score}")
    # print(f"Davies-Bouldin Index: {davies_bouldin_index}")
    # print(f"Calinski-Harabasz Index: {calinski_harabasz_index}")

    if silhouette_score > best_score:
        best_score = silhouette_score
        best_k = k
    else:
        print("Silhouette score is decreasing, stopping search for optimal k.")
        break



final_model = cluster.KMeans(n_clusters=best_k, init='k-means++', n_init=10)
final_labels = final_model.fit_predict(datanp)

plt.scatter(f0, f1, c=final_labels, s=8)
centroids = final_model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title(f"Données après clustering : {name} - Nb clusters = {best_k}")
plt.show()

plt.figure(figsize=(15, 6))

# Silhouette 
plt.subplot(1, 3, 1)
plt.plot(k_list[:len(silhouette_scores)], silhouette_scores, marker='o')
plt.title('Silhouette Score en fonction du nombre de clusters (k)')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Silhouette Score')

plt.grid()

# # Davies-Bouldin 
# plt.subplot(1, 3, 2)
# plt.plot(k_list, davies_bouldin_indices, marker='o')
# plt.title('Davies-Bouldin Index en fonction du nombre de clusters (k)')
# plt.xlabel('Nombre de clusters (k)')
# plt.ylabel('Davies-Bouldin Index')
# plt.xticks(k_list)
# plt.grid()

# # Calinski-Harabasz
# plt.subplot(1, 3, 3)
# plt.plot(k_list, calinski_harabasz_indices, marker='o')
# plt.title('Calinski-Harabasz Index en fonction du nombre de clusters (k)')
# plt.xlabel('Nombre de clusters (k)')
# plt.ylabel('Calinski-Harabasz Index')
# plt.xticks(k_list)
# plt.grid()

plt.tight_layout()
plt.show()


print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
print("labels", labels)

#PLOT INERTIE PAR RAPPORT AU K
# plt.figure(figsize=(10, 6))
# plt.plot(k_list, inertie_list, marker='o')
# plt.title('Inertie en fonction du nombre de clusters (k)')
# plt.xlabel('Nombre de clusters (k)')
# plt.ylabel('Inertie')
# plt.xticks(k_list)
# plt.grid()
# plt.show()


from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)

