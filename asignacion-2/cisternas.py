import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

# 1. Cargar datos
df = pd.read_csv("datoscisterna.csv", parse_dates=['Time'])

# 2. Seleccionar características relevantes para clustering
# Usaremos solo columnas numéricas sin nulos: 'nivel' y 'presion'
df_selected = df[['nivel', 'presion']].dropna()

# 3. Escalado de datos
scaler = StandardScaler()
X = scaler.fit_transform(df_selected)

# 4. K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# 5. Fuzzy C-Means clustering
X_fuzzy = X.T  # Transponer para usar con skfuzzy
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_fuzzy, c=3, m=2, error=0.005, maxiter=1000, init=None
)

# Convertir a etiquetas duras (según máxima pertenencia)
fuzzy_labels = np.argmax(u, axis=0)

# 6. Visualización comparativa
plt.figure(figsize=(12, 5))

# K-Means
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title("Clustering Normal (K-Means)")
plt.xlabel("Nivel (escalado)")
plt.ylabel("Presión (escalado)")

# Fuzzy C-Means
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=fuzzy_labels, cmap='viridis', s=50)
plt.title("Clustering Difuso (Fuzzy C-Means)")
plt.xlabel("Nivel (escalado)")
plt.ylabel("Presión (escalado)")

plt.tight_layout()
plt.show()

# 7. Análisis simple
print("Distribución K-Means:", np.bincount(kmeans_labels))
print("Distribución Fuzzy C-Means:", np.bincount(fuzzy_labels))
