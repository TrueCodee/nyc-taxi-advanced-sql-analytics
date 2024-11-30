# NYC Taxi Data Analysis and Insights

## Overview
This project provides a comprehensive analysis of NYC yellow taxi trip data from January 2015, uncovering patterns and insights related to trip durations, fare amounts, passenger behavior, and location trends. Using PySpark and Spark SQL on Databricks, the project applies advanced data processing and analytics techniques to derive actionable insights.

## Dataset
- **Source**: [NYC Yellow Taxi Trip Data (January 2015) on Kaggle](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data)
- **Description**: This dataset contains details about yellow taxi trips, including trip times, distances, fares, passenger counts, and pickup/drop-off locations.
-  **Key Features**:
  - `tpep_pickup_datetime` & `tpep_dropoff_datetime`: Timestamps of when the trip started and ended
  - `trip_distance`: Distance of the trip in miles
  - `fare_amount`: Fare charged for the trip
  - `passenger_count`: Number of passengers on the trip
  - `pickup_longitude` & `pickup_latitude`: GPS coordinates of pickup location
  - `dropoff_longitude` & `dropoff_latitude`: GPS coordinates of drop-off location
    
## Objectives
- `Outlier Detection`: Identify and analyze outliers in fare amounts.
- `Correlation Analysis`: Explore relationships between key numerical features like fare amounts, trip distances, and total amounts.
- `Trip Duration Prediction`: Analyze trip durations and their relationship with passenger counts and distances.
- `Fare Clustering`: Group trips based on distance and analyze fare distributions within each group.
- `Passenger Behavior`: Understand the impact of passenger counts on fares.
- `Demand and Revenue Trends`: Identify the busiest times, days, and locations, along with revenue patterns.
- `Geographical Hotspots`: Visualize and analyze the most frequent pickup and drop-off locations.
- `Advanced Visualizations`: Use heatmaps, bar charts, and scatter plots to present findings.
  
# Key Insights
- `Outliers`:
Trips with fares over $1,000 and those with zero or negative amounts were identified and excluded from further analysis.
- `Fare vs. Distance`:
Fares increase predictably with distance. For example, trips under 1 mile average $5.34, while trips over 5 miles average $31.28.
- `Passenger Impact`:
Single-passenger trips dominate the dataset. However, larger groups tend to pay higher fares, reflecting either longer distances or larger vehicles.
- `Peak Demand`:
Fridays and Saturdays generate the most revenue, with peak demand hours between 6 PM and 10 PM.
- `Hotspots`:
Manhattan and JFK Airport are major hubs for pickups and drop-offs.
- `Geographic Trends`:
Heatmaps reveal concentrated activity in Manhattan, particularly around high-traffic areas like Times Square.
- `Trip Duration and Distance`:
Longer trips naturally take more time, with a positive correlation between distance and duration.

## How to Run
### Prerequisites
1. **Databricks**: Set up a Databricks environment to run PySpark code. Alternatively, use a local PySpark setup.
2. **Required Libraries**:
   - `PySpark`
   - `Pandas`
   - `Matplotlib`
   - `Seaborn` (for visualizations)
   - `Folium` (for map visualizations)
   - 
### Steps
1. Clone this repository:
```bash
   git clone https://github.com/YourUsername/nyc-taxi-analysis.git
```
2. Upload the files (nyc-taxi-analysis.sql and nyc-taxi-analysis.ipynb) to Databricks or your local environment.
3. Run the SQL scripts and notebooks sequentially to reproduce the analysis.
   
## Project Structure
- `nyc-taxi-analysis.ipynb`: Jupyter Notebook with Python-based analysis and visualizations.
- `nyc-taxi-analysis.sql`: SQL queries for advanced analytics.
- `visualizations`: Contains generated heatmaps, bar charts, and scatter plots.
- `README.md`: Project overview and instructions.

## Summary of Findings
- **Revenue Trends**: Fridays and Saturdays account for the highest revenue, with evening hours being the busiest.
- **Fare Distribution**: Short trips (< 1 mile) have the lowest fares, while trips exceeding 5 miles contribute significantly to total revenue.
- **Hotspot Activity**: Manhattan and JFK Airport emerge as critical hotspots for taxi services.
- **Passenger Trends**: Single-passenger trips dominate the data, but larger groups exhibit higher fare averages.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
