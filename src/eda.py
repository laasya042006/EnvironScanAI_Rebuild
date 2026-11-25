
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_pollution_data.csv")

# Convert date column into datetime format
# Parse dates from the cleaned CSV (YYYY-MM-DD)
df["date"] = pd.to_datetime(
    df["date"],
    format="%Y-%m-%d",
    errors="coerce"
)

print("\nDate quality check:")
print("Total records:", len(df))
print("Valid dates:", df["date"].notna().sum())
print("Invalid dates:", df["date"].isna().sum())
print("\nDate quality check:")
print("Total records:", len(df))
print("Valid dates:", df["date"].notna().sum())
print("Invalid dates:", df["date"].isna().sum())

print("\nSample date values:")
print(df["date"].dropna().head(10))


print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Basic information
print("\n--- Dataset Information ---")
print(df.info())

# 3. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# Create folders for graphs and reports
os.makedirs("reports/figures", exist_ok=True)

# 4. Average pollution by city
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

city_pollution = df.groupby("city")[pollutants].mean()

print("\n--- Average Pollution by City ---")
print(city_pollution.round(2))

city_pollution.to_csv("reports/city_pollution.csv")

# 5. Average pollutant levels
pollution_avg = df[pollutants].mean().sort_values(
    ascending=False
)

print("\n--- Average Pollutant Levels ---")
print(pollution_avg.round(2))

plt.figure(figsize=(10, 5))
pollution_avg.plot(kind="bar")
plt.title("Average Levels of Different Pollutants")
plt.xlabel("Pollutants")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/figures/pollutant_average.png")
plt.close()

# 6. Yearly pollution trends
df["year"] = df["date"].dt.year

yearly_pollution = df.groupby("year")[pollutants].mean()

print("\n--- Yearly Pollution Trends ---")
print(yearly_pollution.round(2))

yearly_pollution.to_csv("reports/yearly_pollution.csv")

plt.figure(figsize=(12, 6))
for pollutant in ["PM2.5", "PM10", "NO2"]:
    plt.plot(
        yearly_pollution.index,
        yearly_pollution[pollutant],
        marker="o",
        label=pollutant
    )

plt.title("Yearly Air Pollution Trends")
plt.xlabel("Year")
plt.ylabel("Average Pollution Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/figures/yearly_pollution.png")
plt.close()

# 7. Correlation between numerical variables
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png")
plt.close()

print("\nPhase 2 completed successfully!")
print("Graphs saved in reports/figures/")
print("Analysis reports saved in reports/")
