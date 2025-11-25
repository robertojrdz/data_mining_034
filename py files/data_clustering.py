import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv('./csv/steam_games_clean.csv')

def parse_owners_midpoint(owners_str):
    try:
        if pd.isna(owners_str):
            return np.nan
        parts = owners_str.split('-')
        min_val = int(parts[0].replace(',', ''))
        max_val = int(parts[1].replace(',', ''))
        return (min_val + max_val) / 2
    except:
        return np.nan

df['owners_numeric'] = df['owners'].apply(parse_owners_midpoint)
df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
df['positive_ratio'] = df['positive_ratings'] / df['total_ratings']
df_clean = df.dropna(subset=['positive_ratings', 'negative_ratings', 'average_playtime','owners_numeric', 'price', 'total_ratings', 'positive_ratio'])

output_file = open('data_clustering_results.txt', 'w')
output_file.write("=" * 80 + '\n')
output_file.write("K-MEANS CLUSTERING ANALYSIS - STEAM GAMES DATASET\n")
output_file.write("=" * 80 + '\n')
output_file.write(f"\nTotal games analyzed: {len(df_clean)} games\n\n")

# ============================================================================
# CLUSTERING MODEL 1: GAME CHARACTERISTICS
output_file.write("=" * 80 + '\n')
output_file.write("CLUSTERING MODEL 1: GAME CHARACTERISTICS\n")
output_file.write("=" * 80 + '\n')
output_file.write("\nClustering based on: ratings, playtime, and price\n\n")
features_1 = ['positive_ratings', 'negative_ratings', 'average_playtime', 'price']
X_1 = df_clean[features_1]
scaler_1 = StandardScaler()
X_1_scaled = scaler_1.fit_transform(X_1)
k_range = range(2, 11)
inertias_1 = []
silhouette_scores_1 = []
calinski_scores_1 = []
davies_bouldin_scores_1 = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_1_scaled)
    inertias_1.append(kmeans.inertia_)
    silhouette_scores_1.append(silhouette_score(X_1_scaled, kmeans.labels_))
    calinski_scores_1.append(calinski_harabasz_score(X_1_scaled, kmeans.labels_))
    davies_bouldin_scores_1.append(davies_bouldin_score(X_1_scaled, kmeans.labels_))

output_file.write("K\tInertia\t\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin\n")
output_file.write("-" * 80 + '\n')
for i, k in enumerate(k_range):
    output_file.write(f"{k}\t{inertias_1[i]:.2f}\t\t{silhouette_scores_1[i]:.4f}\t\t{calinski_scores_1[i]:.2f}\t\t{davies_bouldin_scores_1[i]:.4f}\n")

optimal_k_1 = k_range[np.argmax(silhouette_scores_1)]
output_file.write(f"\nOptimal number of clusters: {optimal_k_1}\n")
output_file.write(f"Selected based on highest Silhouette Score: {max(silhouette_scores_1):.4f}\n")
kmeans_1 = KMeans(n_clusters=optimal_k_1, random_state=42, n_init=10)
clusters_1 = kmeans_1.fit_predict(X_1_scaled)
df_clean['cluster_characteristics'] = clusters_1

for i in range(optimal_k_1):
    cluster_data = df_clean[df_clean['cluster_characteristics'] == i]
    output_file.write(f"\nCluster {i}: {len(cluster_data)} games ({len(cluster_data)/len(df_clean)*100:.1f}%)\n")

# ============================================================================
# CLUSTERING MODEL 2: POPULARITY AND QUALITY
output_file.write("\n" + "=" * 80 + '\n')
output_file.write("CLUSTERING MODEL 2: POPULARITY AND QUALITY\n")
output_file.write("=" * 80 + '\n')
output_file.write("\nClustering based on: ratings, positive ratio, and owners\n\n")
features_2 = ['total_ratings', 'positive_ratio', 'owners_numeric', 'average_playtime']
X_2 = df_clean[features_2]
scaler_2 = StandardScaler()
X_2_scaled = scaler_2.fit_transform(X_2)
inertias_2 = []
silhouette_scores_2 = []
calinski_scores_2 = []
davies_bouldin_scores_2 = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_2_scaled)
    inertias_2.append(kmeans.inertia_)
    silhouette_scores_2.append(silhouette_score(X_2_scaled, kmeans.labels_))
    calinski_scores_2.append(calinski_harabasz_score(X_2_scaled, kmeans.labels_))
    davies_bouldin_scores_2.append(davies_bouldin_score(X_2_scaled, kmeans.labels_))

output_file.write("K\tInertia\t\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin\n")
output_file.write("-" * 80 + '\n')
for i, k in enumerate(k_range):
    output_file.write(f"{k}\t{inertias_2[i]:.2f}\t\t{silhouette_scores_2[i]:.4f}\t\t{calinski_scores_2[i]:.2f}\t\t{davies_bouldin_scores_2[i]:.4f}\n")

optimal_k_2 = k_range[np.argmax(silhouette_scores_2)]
output_file.write(f"\nOptimal number of clusters: {optimal_k_2}\n")
output_file.write(f"Selected based on highest Silhouette Score: {max(silhouette_scores_2):.4f}\n")
kmeans_2 = KMeans(n_clusters=optimal_k_2, random_state=42, n_init=10)
clusters_2 = kmeans_2.fit_predict(X_2_scaled)
df_clean['cluster_popularity'] = clusters_2

for i in range(optimal_k_2):
    cluster_data = df_clean[df_clean['cluster_popularity'] == i]
    output_file.write(f"\nCluster {i}: {len(cluster_data)} games ({len(cluster_data)/len(df_clean)*100:.1f}%)\n")

# ============================================================================
# CLUSTERING MODEL 3: COMPREHENSIVE GAME PROFILE
output_file.write("\n" + "=" * 80 + '\n')
output_file.write("CLUSTERING MODEL 3: COMPREHENSIVE GAME PROFILE\n")
output_file.write("=" * 80 + '\n')
output_file.write("\nClustering based on: all available features\n\n")
features_3 = ['positive_ratings', 'negative_ratings', 'average_playtime', 'price', 'owners_numeric', 'positive_ratio']
X_3 = df_clean[features_3]
scaler_3 = StandardScaler()
X_3_scaled = scaler_3.fit_transform(X_3)
inertias_3 = []
silhouette_scores_3 = []
calinski_scores_3 = []
davies_bouldin_scores_3 = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_3_scaled)
    inertias_3.append(kmeans.inertia_)
    silhouette_scores_3.append(silhouette_score(X_3_scaled, kmeans.labels_))
    calinski_scores_3.append(calinski_harabasz_score(X_3_scaled, kmeans.labels_))
    davies_bouldin_scores_3.append(davies_bouldin_score(X_3_scaled, kmeans.labels_))

output_file.write("K\tInertia\t\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin\n")
output_file.write("-" * 80 + '\n')

for i, k in enumerate(k_range):
    output_file.write(f"{k}\t{inertias_3[i]:.2f}\t\t{silhouette_scores_3[i]:.4f}\t\t{calinski_scores_3[i]:.2f}\t\t{davies_bouldin_scores_3[i]:.4f}\n")

optimal_k_3 = k_range[np.argmax(silhouette_scores_3)]
output_file.write(f"\nOptimal number of clusters: {optimal_k_3}\n")
output_file.write(f"Selected based on highest Silhouette Score: {max(silhouette_scores_3):.4f}\n")
kmeans_3 = KMeans(n_clusters=optimal_k_3, random_state=42, n_init=10)
clusters_3 = kmeans_3.fit_predict(X_3_scaled)
df_clean['cluster_comprehensive'] = clusters_3

for i in range(optimal_k_3):
    cluster_data = df_clean[df_clean['cluster_comprehensive'] == i]
    output_file.write(f"\nCluster {i}: {len(cluster_data)} games ({len(cluster_data)/len(df_clean)*100:.1f}%)\n")

# ============================================================================
# CLUSTER INTERPRETATION AND INSIGHTS
output_file.write("\n" + "=" * 80 + '\n')
output_file.write("CLUSTER INTERPRETATION AND INSIGHTS\n")
output_file.write("=" * 80 + '\n')

output_file.write("\nModel 1 - Game Characteristics:\n")
output_file.write(f"  Optimal Clusters: {optimal_k_1}\n")
output_file.write(f"  Silhouette Score: {max(silhouette_scores_1):.4f}\n")
output_file.write("  Interpretation: Groups games by their rating counts, playtime, and price\n")

output_file.write("\nModel 2 - Popularity and Quality:\n")
output_file.write(f"  Optimal Clusters: {optimal_k_2}\n")
output_file.write(f"  Silhouette Score: {max(silhouette_scores_2):.4f}\n")
output_file.write("  Interpretation: Groups games by popularity (total ratings), quality (positive ratio), and player base\n")

output_file.write("\nModel 3 - Comprehensive Profile:\n")
output_file.write(f"  Optimal Clusters: {optimal_k_3}\n")
output_file.write(f"  Silhouette Score: {max(silhouette_scores_3):.4f}\n")
output_file.write("  Interpretation: Groups games by overall profile including all characteristics\n")

df_clean[['appid', 'name', 'cluster_characteristics', 'cluster_popularity', 'cluster_comprehensive']].to_csv('./csv/steam_games_with_clusters.csv', index=False)
output_file.close()

print("\nResults saved to: kmeans_clustering_results.txt")
print("Clustered data saved to: steam_games_with_clusters.csv\n")