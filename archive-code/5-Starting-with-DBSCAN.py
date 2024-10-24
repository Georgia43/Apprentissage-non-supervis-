import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering

path = '../artificial/'
name = "long2.arff"

databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# PLOT des données initiales en 2D
print("---------------------------------------")
print("Affichage données initiales " + str(name))
f0 = datanp[:, 0]  # Première colonne
f1 = datanp[:, 1]  # Deuxième colonne

plt.scatter(f0, f1, s=8)
plt.title("Données initiales : " + str(name))
plt.show()

# Distances aux k plus proches voisins
# Donnees dans X
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(datanp)
distances, indices = neigh.kneighbors(datanp)

# distance moyenne sur les k plus proches voisins
# en retirant le point " origine "
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(distances.shape[0])])

# trier par ordre croissant
distancetrie = np.sort(newDistances)

plt.title("Plus proches voisins k=" + str(k))
plt.plot(distancetrie)
plt.show()

# Méthode pour trouver le coude : calcul de la pente
differences = np.diff(distancetrie)  # Différences entre points successifs
max_diff = np.argmax(differences)  # Le coude (différence maximale) 

# La valeur optimale de epsilon sera le point correspondant à ce coude
eps_optimal = distancetrie[max_diff]
print(f"Valeur optimale de eps trouvée à l'aide du coude : {eps_optimal}")

####################################################
# Standardisation des données
scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées")
f0_scaled = data_scaled[:, 0]  
f1_scaled = data_scaled[:, 1]  

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Données standardisées")
plt.show()

####################################################
# DBSCAN avec l'epsilon optimal trouvé + itérations sur d'autres valeurs epsilon
min_pts = 5
epsilon_values = [eps_optimal, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# silhouette_scores = []
# davies_bouldin_scores = []
# calinski_harabasz_scores = []
cluster_counts = []

for epsilon in epsilon_values:
    print(f"Appel DBSCAN avec epsilon = {epsilon} et minPts = {min_pts} ... ")
    
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    labels = model.fit_predict(data_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f'Nombre de clusters: {n_clusters}')
    print(f'Nombre de points de bruit: {n_noise}')
   
    cluster_counts.append(n_clusters)
    
    # Si plus d'un cluster, calculer les scores de validation
    # if n_clusters > 1:
    #     silhouette_score = metrics.silhouette_score(data_scaled, labels)
    #     davies_bouldin_score = metrics.davies_bouldin_score(data_scaled, labels)
    #     calinski_harabasz_score = metrics.calinski_harabasz_score(data_scaled, labels)
        
    #     silhouette_scores.append(silhouette_score)
    #     davies_bouldin_scores.append(davies_bouldin_score)
    #     calinski_harabasz_scores.append(calinski_harabasz_score)
    # else:
    #     silhouette_scores.append(None)  # Pas de score valide si 1 ou aucun cluster
    #     davies_bouldin_scores.append(None)
    #     calinski_harabasz_scores.append(None)

    # Visualisation des résultats du clustering pour chaque epsilon
    plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
    plt.title(f"Données après clustering DBSCAN - epsilon={epsilon}")
    plt.show()

# Affichage des scores pour chaque epsilon

# plt.figure(figsize=(15, 5))

# # Silhouette
# plt.subplot(1, 3, 1)
# plt.plot(epsilon_values, silhouette_scores, marker='o')
# plt.title('Silhouette Score vs Epsilon')
# plt.xlabel('Epsilon')
# plt.ylabel('Silhouette Score')
# plt.xticks(epsilon_values)
# plt.grid()

# # Davies-Bouldin 
# plt.subplot(1, 3, 2)
# plt.plot(epsilon_values, davies_bouldin_scores, marker='o')
# plt.title('Davies-Bouldin Score vs Epsilon')
# plt.xlabel('Epsilon')
# plt.ylabel('Davies-Bouldin Score')
# plt.xticks(epsilon_values)
# plt.grid()

# # Calinski-Harabasz 
# plt.subplot(1, 3, 3)
# plt.plot(epsilon_values, calinski_harabasz_scores, marker='o')
# plt.title('Calinski-Harabasz Score vs Epsilon')
# plt.xlabel('Epsilon')
# plt.ylabel('Calinski-Harabasz Score')
# plt.xticks(epsilon_values)
# plt.grid()

# plt.tight_layout()
# plt.show()
