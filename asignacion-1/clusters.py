# Ejemplo de Clustering con K-Means
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos de ejemplo (3 clústeres)
X, y = make_blobs(n_samples=1000, centers=3, cluster_std=0.60, random_state=0)

# Crear el modelo K-Means con 3 clústeres
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Obtener los centroides y etiquetas
centroides = kmeans.cluster_centers_
etiquetas = kmeans.labels_

# Graficar resultados
plt.scatter(X[:, 0], X[:, 1], c=etiquetas, s=50, cmap='viridis')
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=200, alpha=0.75, marker="X", label="Centroides")

plt.title("Ejemplo de Clustering con K-Means")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()
