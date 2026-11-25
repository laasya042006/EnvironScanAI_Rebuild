# EnvironScan AI – Air Pollution Analysis and Clustering

## Project Overview
EnvironScan AI is a data analytics and machine learning project that analyzes air pollution patterns across 10 Indian cities from 2014 to 2024.

The project uses Python, Pandas, Matplotlib, Scikit-learn, and Streamlit to clean pollution data, explore trends, group similar pollution patterns, and visualize findings through an interactive dashboard.

## Objectives
- Clean and preprocess air pollution data.
- Analyze pollutant levels across different cities.
- Study yearly pollution trends.
- Group similar pollution records using K-Means clustering.
- Evaluate clustering performance.
- Display insights through an interactive dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit

## Dataset
The dataset contains 200,900 records from 10 Indian cities, covering 2014–2024.

Pollutants analyzed:
- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

Other features include weather conditions, geographical coordinates, traffic index, and distances from roads, industries, and farms.

## Project Workflow
1. Data collection and inspection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. K-Means clustering
5. Clustering evaluation
6. Interactive dashboard development

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

Run the dashboard:

```bash
streamlit run app.py
```

## Key Findings
- Patna has the highest average PM2.5 and PM10 among the cities analyzed.
- Chennai has the lowest average PM2.5.
- Yearly pollutant averages are relatively stable in this dataset.
- K-Means identifies groups of records with similar pollutant measurements.

## Limitations
- The dataset does not contain verified pollution-source labels.
- Clustering identifies similar pollution patterns, not actual pollution sources.
- Missing fire-distance values are retained and tracked separately.
- Findings describe the supplied dataset and should not be treated as official environmental measurements.

## Future Enhancements
- Add interactive maps.
- Integrate live air-quality data.
- Explore advanced clustering techniques.
- Add more detailed city-wise comparisons.

## Author
EnvironScan AI – Infosys Springboard Project