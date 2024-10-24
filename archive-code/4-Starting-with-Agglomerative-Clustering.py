import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


path = '../artificial/'
name = "flame.arff"

databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])


print("---------------------------------------")
print("Affichage données initiales            " + str(name))
f0 = datanp[:, 0]
f1 = datanp[:, 1]

plt.scatter(f0, f1, s=8)
plt.title("Données initiales : " + str(name))
plt.show()


silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

k_list = range(2, 10)

best_k = None
best_silhouette_score = -1
found_best = False



for k in k_list:
    tps1 = time.time()
    

    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
    labels = model.fit_predict(datanp)
    
    tps2 = time.time()


    silhouette_score = metrics.silhouette_score(datanp, labels)
    # davies_bouldin_score = metrics.davies_bouldin_score(datanp, labels)
    # calinski_harabasz_score = metrics.calinski_harabasz_score(datanp, labels)


    silhouette_scores.append(silhouette_score)
    # davies_bouldin_scores.append(davies_bouldin_score)
    # calinski_harabasz_scores.append(calinski_harabasz_score)

    if silhouette_score > best_silhouette_score:
        best_silhouette_score = silhouette_score
        best_k = k
    else:
        print("Le coefficient de silhouette diminue, arrêt de l'algorithme.")
        found_best = True
        break


    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"Clustering agglomératif (average, n_clusters= {k})")
    plt.show()

    print(f"nb clusters = {k}, runtime = {round((tps2 - tps1) * 1000, 2)} ms")

# metrics_names = ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score']
# scores_values = [silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores]

metrics_names = 'Silhouette Score'
scores_values = silhouette_scores
plt.figure(figsize=(15, 5))
# for i, score in enumerate(scores_values):
#     plt.subplot(1, 3, i + 1)
#     plt.plot(k_list, score, marker='o')
#     plt.title(metrics_names[i])
#     plt.xlabel('Nombre de clusters (k)')
#     plt.ylabel(metrics_names[i])
#     plt.xticks(k_list)
#     plt.grid()

plt.subplot(1,3,1)
plt.plot(k_list[:len(silhouette_scores)], scores_values, marker='o')
plt.title('Silhouette Score vs Nombre de clusters') 
plt.xlabel('Nombre de clusters (k)') 
plt.ylabel('Silhouette Score')

plt.grid()

plt.tight_layout()
plt.show()

seuil_dist = 10
tps1 = time.time()

model = cluster.AgglomerativeClustering(linkage='average', distance_threshold=seuil_dist, n_clusters=None)
labels = model.fit_predict(datanp)

tps2 = time.time()

plt.scatter(f0, f1, c=labels, s=8)
plt.title(f"Clustering agglomératif (average, distance_threshold= {seuil_dist}) " + str(name))
plt.show()

print(f"nb clusters = {model.n_clusters_}, runtime = {round((tps2 - tps1) * 1000, 2)} ms")

