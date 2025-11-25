import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score, f1_score)

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
df['popularity_class'] = pd.qcut(df['total_ratings'], q=3, labels=['Low_Pop', 'Medium_Pop', 'High_Pop'], duplicates='drop')
df_clean = df.dropna(subset=['average_playtime', 'owners_numeric', 'price', 'positive_ratio', 'popularity_class'])

# ============================================================================
# CLASSIFY POPULARITY
features = ['positive_ratio', 'average_playtime', 'price', 'owners_numeric']
X = df_clean[features]
y = df_clean['popularity_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

with open('data_classification_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("    K-NEAREST NEIGHBORS CLASSIFICATION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset size: {len(df_clean)} games\n")
    f.write(f"Train/Test split: 80% / 20%\n")
    f.write(f"K-value range tested: 1-30\n\n")

    f.write("=" * 80 + "\n")
    f.write("    POPULARITY CLASSIFICATION (Low/Medium/High)\n")
    f.write("=" * 80 + "\n")
    f.write(f"Features: {', '.join(features)}\n")
    f.write(f"Optimal k: {optimal_k}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  Accuracy:  {accuracy:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1-Score:  {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n")

print("\nResults exported to: data_classification_results.txt\n")