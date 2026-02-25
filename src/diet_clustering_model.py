import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_pickle("Data/prepro_data/diet_modeling_dataset.pkl")

# Use only nutrient density features
cluster_features = [
    "Sugar_per_1000kcal",
    "Fiber_per_1000kcal",
    "Sodium_per_1000kcal",
    "SatFat_per_1000kcal",
    "Protein_per_1000kcal"
]

X = df[cluster_features]

# Survey weights
weights = df["WTDRD1"]
weights = weights / weights.mean()

# This is the clustering pipeline that will be used in the application
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=20))
])

# Fit model
pipeline.fit(X, kmeans__sample_weight=weights)

# Assign clusters
clusters = pipeline.predict(X)
df["Cluster"] = clusters

# compute weighted cluster distribution for population-level interpretation
cluster_distribution = {}

for c in np.unique(clusters):
    mask = (clusters == c)
    weighted_prop = np.sum(weights[mask]) / np.sum(weights) # sum of weights in cluster / total sum of weights
    cluster_distribution[int(c)] = float(weighted_prop)

cluster_profiles = {}

for c in np.unique(clusters):
    mask = (clusters == c)
    profile = {}
    for feature in cluster_features:
        profile[feature] = float(
            np.average(df.loc[mask, feature], weights=weights[mask])
        )
    cluster_profiles[int(c)] = profile

# save cluster profiles for use in app
artifact = {
    "pipeline": pipeline,
    "cluster_features": cluster_features,
    "cluster_distribution": cluster_distribution,
    "cluster_profiles": cluster_profiles,
    "scaler": pipeline.named_steps["scaler"]
}




# This is for diagnostic purposes to evaluate cluster quality
from sklearn.metrics import silhouette_score

# Get scaled data
X_scaled = pipeline.named_steps["scaler"].transform(X)

sil_score = silhouette_score(X_scaled, clusters)

print(f"Silhouette Score: {sil_score:.3f}")

print("\nRaw Cluster Counts:")
print(df["Cluster"].value_counts().sort_index())

print("\nWeighted Cluster Distribution:")
for c, prop in cluster_distribution.items():
    print(f"Cluster {c}: {prop*100:.1f}%")
# --------------------------------------------------------

joblib.dump(artifact, "Data/prepro_data/diet_clustering_model.pkl")

print("Clustering model saved to diet_clustering_model.pkl")