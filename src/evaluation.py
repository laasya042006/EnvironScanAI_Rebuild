
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_pollution_data.csv")

# Select pollutant columns
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
X = df[pollutants].fillna(df[pollutants].median())

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Evaluate clustering
silhouette = silhouette_score(X_scaled, labels)
db_score = davies_bouldin_score(X_scaled, labels)

print("MODEL EVALUATION RESULTS")
print("------------------------")
print("Number of clusters:", len(set(labels)))
print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Score:", round(db_score, 4))

# Save evaluation results
results = pd.DataFrame({
    "Metric": ["Silhouette Score", "Davies-Bouldin Score"],
    "Value": [silhouette, db_score]
})

results.to_csv("reports/model_evaluation.csv", index=False)

print("\nEvaluation report saved successfully!")
