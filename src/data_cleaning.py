
"""
Phase 1 — Data inspection and cleaning for EnvironScan AI.
Run from the project root:
    python src/data_cleaning.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "unlabeled_pollution_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

NUMERIC_COLUMNS = [
    "latitude", "longitude", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "temp", "humidity", "wind_speed", "wind_dir", "dist_to_road",
    "dist_to_industry", "dist_to_farm", "fire_nearby",
    "fire_min_dist_km", "traffic_index",
]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    original_rows, original_cols = df.shape

    required_columns = {"city", "location_id", "date", *NUMERIC_COLUMNS}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_columns}"
        )

    # Correct date conversion: day-month-year
    df["date"] = pd.to_datetime(
        df["date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    print("\nDate validation:")
    print("Total records:", len(df))
    print("Valid dates:", df["date"].notna().sum())
    print("Invalid dates:", df["date"].isna().sum())

    # Remove exact duplicate records
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Convert numeric columns
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Clean text columns
    df["city"] = df["city"].astype("string").str.strip()
    df["location_id"] = (
        df["location_id"].astype("string").str.strip()
    )

    # Flag missing fire distance
    df["fire_distance_missing"] = (
        df["fire_min_dist_km"].isna().astype("int8")
    )

    # Data validation checks
    invalid_lat = ~df["latitude"].between(-90, 90)
    invalid_lon = ~df["longitude"].between(-180, 180)
    invalid_humidity = ~df["humidity"].between(0, 100)
    invalid_fire_flag = ~df["fire_nearby"].isin([0, 1])

    invalid_counts = {
        "invalid_latitude": int(invalid_lat.sum()),
        "invalid_longitude": int(invalid_lon.sum()),
        "humidity_outside_0_100": int(invalid_humidity.sum()),
        "fire_nearby_not_0_or_1": int(invalid_fire_flag.sum()),
        "unparsed_dates": int(df["date"].isna().sum()),
    }

    # Save cleaned dataset
    output_file = PROCESSED_DIR / "cleaned_pollution_data.csv"
    df.to_csv(output_file, index=False, date_format="%Y-%m-%d")

    # Generate data quality report
    missing = df.isna().sum().sort_values(ascending=False)

    report_lines = [
        "ENVIRONSCAN AI — DATA QUALITY REPORT",
        "=" * 42,
        f"Input file: {RAW_FILE.name}",
        f"Original rows: {original_rows}",
        f"Original columns: {original_cols}",
        f"Rows after duplicate removal: {len(df)}",
        f"Exact duplicates removed: {duplicate_count}",
        f"Unique cities: {df['city'].nunique()}",
        f"Date range: {df['date'].min()} to {df['date'].max()}",
        "",
        "MISSING VALUES BY COLUMN",
        "-" * 28,
    ]

    report_lines.extend(
        f"{name}: {count}" for name, count in missing.items()
    )

    report_lines.extend(["", "VALIDATION COUNTS", "-" * 18])

    report_lines.extend(
        f"{name}: {count}" for name, count in invalid_counts.items()
    )

    report_lines.extend([
        "",
        "NOTE:",
        "The dataset has no verified pollution-source target column.",
        "Do not claim a model predicts real pollution sources until labels",
        "are independently defined and validated.",
        "",
        f"Cleaned file: {output_file.relative_to(ROOT)}",
    ])

    report_file = REPORT_DIR / "data_quality_report.txt"
    report_file.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\nPhase 1 completed successfully!")
    print(f"Original shape: {original_rows:,} rows x {original_cols} columns")
    print(f"Rows after duplicate removal: {len(df):,}")
    print(f"Cities: {df['city'].nunique()}")
    print(f"Missing fire_min_dist_km: {df['fire_min_dist_km'].isna().sum():,}")
    print(f"Saved cleaned data: {output_file}")
    print(f"Saved quality report: {report_file}")


if __name__ == "__main__":
    main()
