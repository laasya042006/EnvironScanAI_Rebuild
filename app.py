
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EnvironScan AI",
    page_icon="🌍",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}
[data-testid="stMetric"] {
    background-color: #1b2635;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #26384d;
}
h1, h2, h3 {
    color: #42e8b4;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv(
        "data/processed/clustered_pollution_data.csv"
    )
    data["date"] = pd.to_datetime(data["date"])
    return data

try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "Clustered dataset not found. Run python src/clustering.py first."
    )
    st.stop()

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 EnvironScan AI")
st.sidebar.caption("Environmental Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "City Analysis",
        "City Comparison",
        "Pollution Trends",
        "Cluster Explorer",
        "Download Reports",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Dashboard Filters")

cities = ["All Cities"] + sorted(data["city"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", cities)

min_date = data["date"].min().date()
max_date = data["date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

df = data[
    (data["date"].dt.date >= start_date) &
    (data["date"].dt.date <= end_date)
].copy()

if selected_city != "All Cities":
    df = df[df["city"] == selected_city]

if df.empty:
    st.warning("No records found for the selected filters.")
    st.stop()

# ---------------- HEADER ----------------
st.title("🌍 EnvironScan AI")
st.markdown(
    "### Smart Air Pollution Analysis & Environmental Monitoring"
)
st.caption(
    "Explore pollution measurements, city comparisons, "
    "historical trends and machine-learning clusters."
)

st.divider()

# ---------------- OVERVIEW ----------------
if page == "Overview":

    st.subheader("📊 Pollution Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Cities Covered", df["city"].nunique())
    c3.metric("Average PM2.5", round(df["PM2.5"].mean(), 2))
    c4.metric("Average PM10", round(df["PM10"].mean(), 2))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Pollutant Average Levels")

        averages = df[pollutants].mean().reset_index()
        averages.columns = ["Pollutant", "Average"]

        fig = px.bar(
            averages,
            x="Pollutant",
            y="Average",
            color="Pollutant",
            template="plotly_dark",
            title="Average Pollutant Measurements"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("City-wise PM2.5")

        city_data = (
            df.groupby("city")["PM2.5"]
            .mean()
            .reset_index()
            .sort_values("PM2.5")
        )

        fig = px.bar(
            city_data,
            x="PM2.5",
            y="city",
            orientation="h",
            color="PM2.5",
            template="plotly_dark",
            title="Average PM2.5 by City"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("💡 Automatic Data Insights")

    highest = df.groupby("city")["PM2.5"].mean().idxmax()
    lowest = df.groupby("city")["PM2.5"].mean().idxmin()

    st.info(
        f"Highest average PM2.5 in the filtered data: {highest}"
    )
    st.success(
        f"Lowest average PM2.5 in the filtered data: {lowest}"
    )

    st.caption(
        "These insights describe the supplied dataset, "
        "not official real-time air-quality conditions."
    )

# ---------------- CITY ANALYSIS ----------------
elif page == "City Analysis":

    st.subheader("🏙️ City Pollution Analysis")

    pollutant = st.selectbox(
        "Choose pollutant",
        pollutants
    )

    city_avg = (
        df.groupby("city")[pollutant]
        .mean()
        .reset_index()
        .sort_values(pollutant, ascending=False)
    )

    fig = px.bar(
        city_avg,
        x="city",
        y=pollutant,
        color=pollutant,
        template="plotly_dark",
        title=f"{pollutant} Measurements Across Cities"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("City Statistics")
    st.dataframe(city_avg, use_container_width=True)

    st.subheader("📍 City Locations")

    locations = (
        df.groupby("city")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            pollution=(pollutant, "mean")
        )
        .reset_index()
    )

    fig = px.scatter_geo(
        locations,
        lat="latitude",
        lon="longitude",
        color="pollution",
        hover_name="city",
        size="pollution",
        scope="asia",
        template="plotly_dark",
        title=f"Geographical Distribution of {pollutant}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CITY COMPARISON ----------------
elif page == "City Comparison":

    st.subheader("⚖️ Compare Two Cities")

    all_cities = sorted(data["city"].unique())

    col1, col2 = st.columns(2)

    with col1:
        city1 = st.selectbox(
            "First City",
            all_cities,
            index=0
        )

    with col2:
        city2 = st.selectbox(
            "Second City",
            all_cities,
            index=min(1, len(all_cities) - 1)
        )

    comparison = data[
        data["city"].isin([city1, city2]) &
        (data["date"].dt.date >= start_date) &
        (data["date"].dt.date <= end_date)
    ]

    comparison_data = (
        comparison.groupby("city")[pollutants]
        .mean()
        .reset_index()
        .melt(
            id_vars="city",
            var_name="Pollutant",
            value_name="Average"
        )
    )

    fig = px.bar(
        comparison_data,
        x="Pollutant",
        y="Average",
        color="city",
        barmode="group",
        template="plotly_dark",
        title="Pollutant Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comparison.groupby("city")[pollutants].mean(),
        use_container_width=True
    )

# ---------------- POLLUTION TRENDS ----------------
elif page == "Pollution Trends":

    st.subheader("📈 Historical Pollution Trends")

    pollutant = st.selectbox(
        "Select pollutant",
        pollutants
    )

    trend_type = st.radio(
        "Trend type",
        ["Yearly", "Monthly"],
        horizontal=True
    )

    if trend_type == "Yearly":
        trend = (
            df.groupby(df["date"].dt.year)[pollutant]
            .mean()
            .reset_index()
        )
        trend.columns = ["Year", "Average"]

        x_axis = "Year"

    else:
        monthly_df = df.copy()
        monthly_df["Month"] = (
            monthly_df["date"].dt.to_period("M").astype(str)
        )

        trend = (
            monthly_df.groupby("Month")[pollutant]
            .mean()
            .reset_index()
        )

        x_axis = "Month"
        trend.columns = ["Month", "Average"]

    fig = px.line(
        trend,
        x=x_axis,
        y="Average",
        markers=True,
        template="plotly_dark",
        title=f"{pollutant} {trend_type} Trend"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(trend, use_container_width=True)

# ---------------- CLUSTER EXPLORER ----------------
elif page == "Cluster Explorer":

    st.subheader("🤖 Machine Learning Cluster Explorer")

    if "pollution_cluster" not in df.columns:
        st.warning("Cluster information is unavailable.")
    else:
        cluster_summary = (
            df.groupby("pollution_cluster")[pollutants]
            .mean()
            .reset_index()
        )

        st.write("Average pollutant measurements in each cluster.")

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        selected_pollutant = st.selectbox(
            "Select pollutant for cluster comparison",
            pollutants
        )

        fig = px.bar(
            cluster_summary,
            x="pollution_cluster",
            y=selected_pollutant,
            color="pollution_cluster",
            template="plotly_dark",
            title=f"{selected_pollutant} by Cluster"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Clusters represent groups of records with similar "
            "pollutant measurements. They do not identify "
            "confirmed pollution sources."
        )

# ---------------- DOWNLOAD REPORTS ----------------
elif page == "Download Reports":

    st.subheader("📥 Download Your Reports")

    st.write("Download the data currently selected in the dashboard.")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="environscan_filtered_data.csv",
        mime="text/csv"
    )

    summary = (
        df.groupby("city")[pollutants]
        .mean()
        .reset_index()
    )

    summary_csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download City Summary (CSV)",
        data=summary_csv,
        file_name="city_pollution_summary.csv",
        mime="text/csv"
    )

    st.success("Your reports are ready to download.")

# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":

    st.subheader("🌱 About EnvironScan AI")

    st.write("""
    EnvironScan AI is an air pollution analysis and machine
    learning project developed using Python.

    It explores pollution measurements across ten Indian cities
    from 2014 to 2024.
    """)

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Matplotlib and Plotly
    - Scikit-learn
    - Streamlit
    """)

    st.markdown("### Machine Learning")

    st.write("""
    K-Means clustering groups records based on similarities
    in pollutant measurements.
    """)

    st.markdown("### Dataset Information")

    st.write(f"Total records: {len(data):,}")
    st.write(f"Cities: {data['city'].nunique()}")
    st.write(
        f"Date range: {data['date'].min().date()} "
        f"to {data['date'].max().date()}"
    )

    st.markdown("### Project Limitations")

    st.write("""
    - Dataset does not contain verified pollution-source labels.
    - Cluster results are pattern groups, not source predictions.
    - Dashboard displays historical dataset measurements,
      not live air-quality readings.
    """)

# ---------------- FOOTER ----------------
st.divider()

st.markdown(
    "<center>🌍 EnvironScan AI | Air Pollution Analysis "
    "and Environmental Intelligence</center>",
    unsafe_allow_html=True
)
