
import pandas as pd

df = pd.read_csv("data/raw/unlabeled_pollution_data.csv")

print("Original date column:")
print(df["date"].head(15))

# Test the expected date format
converted = pd.to_datetime(
    df["date"],
    format="%d-%m-%Y",
    errors="coerce"
)

print("\nDate conversion results:")
print("Total records:", len(df))
print("Successfully converted:", converted.notna().sum())
print("Failed conversions:", converted.isna().sum())

print("\nExamples of failed date values:")
print(df.loc[converted.isna(), "date"].head(20))
