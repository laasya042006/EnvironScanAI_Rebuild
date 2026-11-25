
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_pollution_data.csv")

# Select pollution columns
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
X = df[pollutants].copy()

# Handle missing values
X = X.fillna(X.median())

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["pollution_cluster"] = kmeans.fit_predict(X_scaled)

# Save clustered dataset
df.to_csv(
    "data/processed/clustered_pollution_data.csv",
    index=False
)

# Calculate cluster averages
cluster_summary = df.groupby("pollution_cluster")[pollutants].mean()
cluster_summary.to_csv("reports/cluster_summary.csv")

print("Clustering completed successfully!")
print("\nCluster Summary:")
print(cluster_summary)

# Create visualization
cluster_summary.plot(kind="bar", figsize=(10, 6))
plt.title("Average Pollutant Levels by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Pollutant Value")
plt.tight_layout()
plt.savefig("reports/figures/pollution_clusters.png")
plt.show()
