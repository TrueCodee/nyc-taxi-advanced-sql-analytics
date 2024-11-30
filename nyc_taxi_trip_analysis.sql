-- Databricks notebook source
-- MAGIC %md
-- MAGIC Assignment #2

-- COMMAND ----------

-- MAGIC %python
-- MAGIC %pip install folium

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #load Dataset
-- MAGIC df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/aryxnjain@gmail.com/yellow_tripdata_2015_01.csv")
-- MAGIC # Creating a temporary SQL view
-- MAGIC df.createOrReplaceTempView("tripdata")
-- MAGIC df

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q1: Outlier Detection

-- COMMAND ----------

SELECT *
FROM tripdata
WHERE CAST(fare_amount AS DOUBLE) > 1000
ORDER BY CAST(fare_amount AS DOUBLE) DESC

-- COMMAND ----------

SELECT *
FROM tripdata
WHERE CAST(fare_amount AS DOUBLE) <= 0
ORDER BY CAST(fare_amount AS DOUBLE) ASC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q2: Correlation Analysis

-- COMMAND ----------

-- correlation between fare amount, total amount, and trip distance for trips with positive values
SELECT 
    CORR(CAST(fare_amount AS DOUBLE), CAST(trip_distance AS DOUBLE)) AS fare_distance_correlation,
    CORR(CAST(total_amount AS DOUBLE), CAST(trip_distance AS DOUBLE)) AS total_distance_correlation
FROM tripdata
WHERE 
    CAST(fare_amount AS DOUBLE) > 0 
    AND CAST(total_amount AS DOUBLE) > 0 
    AND CAST(trip_distance AS DOUBLE) > 0

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Fare Amount vs. Trip Distance:
-- MAGIC
-- MAGIC The correlation between fare amount and trip distance was around 0.00046, suggesting that how far a taxi travels doesn’t strongly affect the fare. This is surprising since we’d expect longer trips to have higher fares. This weak correlation could mean other factors, like base charges, surcharges, or even traffic conditions, have more influence on the fare than the actual distance traveled.
-- MAGIC Total Amount vs. Trip Distance:
-- MAGIC
-- MAGIC The correlation between total amount and trip distance was even smaller at 0.0000339, implying that the total cost (including tips, tolls, and surcharges) also doesn’t depend much on how far the trip is. This suggests that fixed costs, like base fare or additional charges, might make the trip distance less important in determining the final price.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q3: Trip Duration Prediction

-- COMMAND ----------

-- calculate the average trip duration (in minutes) grouped by passenger count
SELECT 
    CAST(passenger_count AS INT) AS passenger_count,
    AVG(
        (UNIX_TIMESTAMP(CAST(tpep_dropoff_datetime AS TIMESTAMP)) - 
         UNIX_TIMESTAMP(CAST(tpep_pickup_datetime AS TIMESTAMP))) / 60
    ) AS avg_duration_minutes
FROM tripdata
WHERE CAST(passenger_count AS INT) > 0
GROUP BY CAST(passenger_count AS INT)
ORDER BY CAST(passenger_count AS INT)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q4: Trip Clustering

-- COMMAND ----------

-- Categorize trips into distance bins and calculate the average fare for each bin, ordered by distance
WITH distance_bins AS (
    SELECT 
        CASE
            WHEN trip_distance < 1 THEN '<1 mile'
            WHEN trip_distance >= 1 AND trip_distance <= 2 THEN '1-2 miles'
            WHEN trip_distance > 2 AND trip_distance <= 5 THEN '2-5 miles'
            ELSE '>5 miles'
        END AS distance_bin,
        CAST(fare_amount AS DOUBLE) AS fare_amount,
        CASE
            WHEN trip_distance < 1 THEN 0
            WHEN trip_distance >= 1 AND trip_distance <= 2 THEN 1
            WHEN trip_distance > 2 AND trip_distance <= 5 THEN 2
            ELSE 3
        END AS distance_order
    FROM tripdata
)
SELECT 
    distance_bin,
    AVG(fare_amount) AS avg_fare_amount
FROM distance_bins
GROUP BY distance_bin, distance_order
ORDER BY distance_order

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q5: Fare Amount vs. Distance Analysis

-- COMMAND ----------

-- Categorize trips into distance bins and calculate the average fare for each bin, ordered logically by distance
SELECT 
    CASE 
        WHEN CAST(trip_distance AS DOUBLE) < 1 THEN '< 1 mile'
        WHEN CAST(trip_distance AS DOUBLE) < 2 THEN '1-2 miles'
        WHEN CAST(trip_distance AS DOUBLE) < 3 THEN '2-3 miles'
        WHEN CAST(trip_distance AS DOUBLE) < 4 THEN '3-4 miles'
        WHEN CAST(trip_distance AS DOUBLE) < 5 THEN '4-5 miles'
        ELSE '> 5 miles'
    END AS distance_bin,
    AVG(CAST(fare_amount AS DOUBLE)) AS avg_fare_amount
FROM tripdata
WHERE CAST(fare_amount AS DOUBLE) > 0 AND CAST(trip_distance AS DOUBLE) > 0
GROUP BY 
    CASE 
        WHEN CAST(trip_distance AS DOUBLE) < 1 THEN '< 1 mile'
        WHEN CAST(trip_distance AS DOUBLE) < 2 THEN '1-2 miles'
        WHEN CAST(trip_distance AS DOUBLE) < 3 THEN '2-3 miles'
        WHEN CAST(trip_distance AS DOUBLE) < 4 THEN '3-4 miles'
        WHEN CAST(trip_distance AS DOUBLE) < 5 THEN '4-5 miles'
        ELSE '> 5 miles'
    END
ORDER BY 
    CASE distance_bin
        WHEN '< 1 mile' THEN 1
        WHEN '1-2 miles' THEN 2
        WHEN '2-3 miles' THEN 3
        WHEN '3-4 miles' THEN 4
        WHEN '4-5 miles' THEN 5
        ELSE 6
    END

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Findings show that the average fare amount increases as the distance increases.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q6: Passenger Count Distribution

-- COMMAND ----------

-- Number of trips and Average fare amount for each passenger count
SELECT 
    CAST(passenger_count AS INT) AS passenger_count,
    COUNT(*) AS trip_count,
    AVG(CAST(fare_amount AS DOUBLE)) AS avg_fare_amount
FROM tripdata
WHERE CAST(passenger_count AS INT) > 0 
  AND CAST(fare_amount AS DOUBLE) > 0
GROUP BY CAST(passenger_count AS INT)
ORDER BY CAST(passenger_count AS INT)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q7 - Heatmap of Trip Frequencies

-- COMMAND ----------

-- Most frequent pickup locations 
SELECT 
    ROUND(CAST(pickup_latitude AS DOUBLE), 3) AS pickup_latitude,
    ROUND(CAST(pickup_longitude AS DOUBLE), 3) AS pickup_longitude,
    COUNT(*) AS trip_count
FROM tripdata
WHERE 
    CAST(pickup_latitude AS DOUBLE) IS NOT NULL
    AND CAST(pickup_longitude AS DOUBLE) IS NOT NULL
GROUP BY 
    ROUND(CAST(pickup_latitude AS DOUBLE), 3), 
    ROUND(CAST(pickup_longitude AS DOUBLE), 3)
ORDER BY trip_count DESC
LIMIT 10;  -- top 10 only


-- COMMAND ----------

-- Most frequent drop off locations 
SELECT 
    ROUND(CAST(dropoff_latitude AS DOUBLE), 3) AS dropoff_latitude,
    ROUND(CAST(dropoff_longitude AS DOUBLE), 3) AS dropoff_longitude,
    COUNT(*) AS trip_count
FROM tripdata
WHERE 
    CAST(dropoff_latitude AS DOUBLE) IS NOT NULL
    AND CAST(dropoff_longitude AS DOUBLE) IS NOT NULL
GROUP BY 
    ROUND(CAST(dropoff_latitude AS DOUBLE), 3), 
    ROUND(CAST(dropoff_longitude AS DOUBLE), 3)
ORDER BY trip_count DESC
LIMIT 10;  -- top 10 only


-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC import seaborn as sns
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC pickup_data = {
-- MAGIC     'pickup_latitude': [0,40.751,40.75,40.645,40.75,40.774,40.774,40.769,40.645,40.752],
-- MAGIC     'pickup_longitude': [0,-73.994,-73.991,-73.782,-73.992,-73.871,-73.873,-73.863,-73.777,-73.978],
-- MAGIC     'trip_count': [243478,61880,58728,53204,49092,47972,43382,39928,36112,35927]
-- MAGIC }
-- MAGIC
-- MAGIC dropoff_data = {
-- MAGIC     'dropoff_latitude': [0,40.75,40.749,40.75,40.751,40.75,40.757,40.762,40.774,40.756],
-- MAGIC     'dropoff_longitude': [0,-73.991,-73.992,-73.995,-73.991,-73.992,-73.99,-73.979,-73.871,-73.991],
-- MAGIC     'trip_count': [235318,58666,45459,41244,40824,25570,24678,23961,22701,21301]
-- MAGIC }
-- MAGIC
-- MAGIC pickup_df = pd.DataFrame(pickup_data)
-- MAGIC dropoff_df = pd.DataFrame(dropoff_data)
-- MAGIC
-- MAGIC #heatmap pickup
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC sns.heatmap(pickup_df.pivot("pickup_latitude", "pickup_longitude", "trip_count"), cmap="Reds", annot=True)
-- MAGIC plt.title("Heatmap of Pickup Locations")
-- MAGIC plt.show()
-- MAGIC
-- MAGIC #heatmap dropoff
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC sns.heatmap(dropoff_df.pivot("dropoff_latitude", "dropoff_longitude", "trip_count"), cmap="Blues", annot=True)
-- MAGIC plt.title("Heatmap of Drop-off Locations")
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q8 - Busiest Days and Times Analysis

-- COMMAND ----------

-- Revenue each day
SELECT 
    DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS day_of_week,
    SUM(CAST(fare_amount AS DOUBLE)) AS total_revenue
FROM tripdata
WHERE CAST(fare_amount AS DOUBLE) > 0
GROUP BY DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP))
ORDER BY total_revenue DESC;



-- COMMAND ----------

-- Busiest hour of the day based on number of pickups
SELECT 
    HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS hour_of_day,
    COUNT(*) AS pickup_count
FROM tripdata
GROUP BY HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP))
ORDER BY pickup_count DESC;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Findings show that the busiest times are 5-10 PM, with the busiest time being 7 PM. 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q9 - Trip Duration and Time of Day Analysis

-- COMMAND ----------

-- Average trip duration for each hour of the day
SELECT 
    HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS hour_of_day,
    AVG(
        (UNIX_TIMESTAMP(CAST(tpep_dropoff_datetime AS TIMESTAMP)) - 
         UNIX_TIMESTAMP(CAST(tpep_pickup_datetime AS TIMESTAMP))) / 60
    ) AS avg_duration_minutes
FROM tripdata
WHERE 
    CAST(tpep_pickup_datetime AS TIMESTAMP) IS NOT NULL
    AND CAST(tpep_dropoff_datetime AS TIMESTAMP) IS NOT NULL
GROUP BY HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP))
ORDER BY hour_of_day;


-- COMMAND ----------

-- MAGIC %python
-- MAGIC data = {
-- MAGIC     'hour_of_day': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
-- MAGIC     'avg_duration_minutes': [13.313322275913483,13.054973602331446,13.145008571616827,13.435592589980732,13.242055614883688,13.76538367977902,11.877530064008267,12.796801292914775,13.587299440016215,13.489896511813681,13.228424923713806,13.17238671771968,13.280298749710424,13.579905845567446,14.454319127053141,32.09129095959835,14.041422302308805,13.826888584857201,13.33810871529094,12.883962532444134,12.333422598934755,12.416918945987238,12.678930717359632,13.098988655180626]
-- MAGIC }
-- MAGIC
-- MAGIC df_data = pd.DataFrame(data)
-- MAGIC
-- MAGIC #plot
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.plot(df_data['hour_of_day'], df_data['avg_duration_minutes'], marker='o', color='b')
-- MAGIC plt.title('Average Trip Duration by Hour of the Day')
-- MAGIC plt.xlabel('Hour of the Day')
-- MAGIC plt.ylabel('Average Trip Duration (Minutes)')
-- MAGIC plt.xticks(df_data['hour_of_day'])  # Show all hours
-- MAGIC plt.grid(True)
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC From the graph, we learned that the trip duration peaks at 3 PM, and is at its lowest at 6 AM, but doesn’t vary greatly outside of those two.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q10 - Payment Type Fare Comparison

-- COMMAND ----------

-- Average fare amounts by payment type.
SELECT 
    payment_type,
    COUNT(*) AS trip_count,
    AVG(CAST(fare_amount AS DOUBLE)) AS avg_fare_amount
FROM tripdata
WHERE CAST(fare_amount AS DOUBLE) > 0
GROUP BY payment_type
ORDER BY avg_fare_amount DESC;



-- COMMAND ----------

-- MAGIC %python
-- MAGIC # bar chart comparing average fares for each payment type.
-- MAGIC data = {
-- MAGIC     'payment_type': [4,1,3,2,5],
-- MAGIC     'avg_fare_amount': [13.389695041906496,12.502455255685902,11.40895290980564,10.956297484636094,6]
-- MAGIC }
-- MAGIC
-- MAGIC df_data = pd.DataFrame(data)
-- MAGIC
-- MAGIC #plot
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.bar(df_data['payment_type'], df_data['avg_fare_amount'], color='skyblue')
-- MAGIC plt.title('Average Fare Amount by Payment Type')
-- MAGIC plt.xlabel('Payment Type')
-- MAGIC plt.ylabel('Average Fare Amount ($)')
-- MAGIC plt.xticks(rotation=45)  # Rotate x-axis labels for readability
-- MAGIC plt.grid(True, axis='y')
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q11 - Time Series Analysis of Trips

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import to_timestamp, col
-- MAGIC
-- MAGIC # Parsing 'tpep_pickup_datetime' as a timestamp
-- MAGIC df_with_date = df.withColumn("trip_date", to_timestamp(col("tpep_pickup_datetime"), "yyyy-MM-dd HH:mm:ss"))
-- MAGIC
-- MAGIC # Creating a temporary view with the parsed timestamp
-- MAGIC df_with_date.createOrReplaceTempView("tripdata_with_date")

-- COMMAND ----------

SELECT trip_date, COUNT(*) AS trip_count
FROM tripdata_with_date
GROUP BY trip_date
ORDER BY trip_date

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Running the SQL query and converting the results to Pandas
-- MAGIC trip_counts_per_day = spark.sql("""
-- MAGIC     SELECT DATE(trip_date) AS trip_day, COUNT(*) AS trip_count
-- MAGIC     FROM tripdata_with_date
-- MAGIC     GROUP BY DATE(trip_date)
-- MAGIC     ORDER BY trip_day
-- MAGIC """).toPandas()
-- MAGIC
-- MAGIC # Plotting the data
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.plot(trip_counts_per_day["trip_day"], trip_counts_per_day["trip_count"], marker="o", linestyle="-")
-- MAGIC plt.title("Trip Counts Per Day")
-- MAGIC plt.xlabel("Date")
-- MAGIC plt.ylabel("Number of Trips")
-- MAGIC plt.xticks(rotation=45)
-- MAGIC plt.grid(True)
-- MAGIC plt.tight_layout()
-- MAGIC
-- MAGIC # Show the plot
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Comments about trends or significant spikes.
-- MAGIC The graph shows us that trip counts peaked on the 10th and 31st, and were at their lowest on the 26th and 27th. Overall the trip counts were the most consistently high in the middle of the month from the 8th to the 24th.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q12 - Location Analysis

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import struct, col
-- MAGIC
-- MAGIC # Parsing 'tpep_pickup_datetime' as a timestamp
-- MAGIC df_with_locations = df.withColumn("pickup_location",struct(df.pickup_longitude,df.pickup_latitude)).withColumn("dropoff_location",struct(df.dropoff_longitude,df.dropoff_latitude))
-- MAGIC
-- MAGIC df_with_locations = df_with_locations.filter(
-- MAGIC     (col("pickup_location.pickup_longitude") != 0) | (col("pickup_location.pickup_latitude") != 0)
-- MAGIC ).filter(
-- MAGIC     (col("dropoff_location.dropoff_longitude") != 0) | (col("dropoff_location.dropoff_latitude") != 0)
-- MAGIC )
-- MAGIC
-- MAGIC df_with_locations.createOrReplaceTempView("df_with_locations")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Query to get top pickup locations
-- MAGIC top_pickup_locations = spark.sql("""
-- MAGIC     SELECT pickup_location, COUNT(*) as count
-- MAGIC     FROM df_with_locations
-- MAGIC     GROUP BY pickup_location
-- MAGIC     ORDER BY count DESC
-- MAGIC     LIMIT 10
-- MAGIC """)
-- MAGIC
-- MAGIC # Query to get top dropoff locations
-- MAGIC top_dropoff_locations = spark.sql("""
-- MAGIC     SELECT dropoff_location, COUNT(*) as count
-- MAGIC     FROM df_with_locations
-- MAGIC     GROUP BY dropoff_location
-- MAGIC     ORDER BY count DESC
-- MAGIC     LIMIT 10
-- MAGIC """)
-- MAGIC
-- MAGIC # Displaying tables in SparkSQL
-- MAGIC top_pickup_locations.show()
-- MAGIC top_dropoff_locations.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import folium
-- MAGIC from pyspark.sql import functions as F
-- MAGIC
-- MAGIC # Calculate mean latitude and longitude for centering the map
-- MAGIC mean_coords = df_with_locations.select(
-- MAGIC     F.avg("pickup_location.pickup_latitude").alias("mean_latitude"),
-- MAGIC     F.avg("pickup_location.pickup_longitude").alias("mean_longitude")
-- MAGIC ).collect()[0]
-- MAGIC
-- MAGIC mean_latitude = mean_coords["mean_latitude"]
-- MAGIC mean_longitude = mean_coords["mean_longitude"]
-- MAGIC
-- MAGIC # Get top pickup and dropoff locations in PySpark
-- MAGIC top_pickup_locations_list = top_pickup_locations.collect()
-- MAGIC top_dropoff_locations_list = top_dropoff_locations.collect()
-- MAGIC
-- MAGIC # Initialize the map centered on mean coordinates
-- MAGIC nyc_map = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=12)
-- MAGIC
-- MAGIC # Plot top pickup locations in blue
-- MAGIC for location in top_pickup_locations_list:
-- MAGIC     pickup_location = location['pickup_location']
-- MAGIC     folium.CircleMarker(
-- MAGIC         location=[pickup_location[1], pickup_location[0]],  # (latitude, longitude)
-- MAGIC         radius=5,
-- MAGIC         color='blue',
-- MAGIC         fill=True,
-- MAGIC         fill_color='blue',
-- MAGIC         fill_opacity=0.6,
-- MAGIC         popup=f"Pickup Count: {location['count']}"
-- MAGIC     ).add_to(nyc_map)
-- MAGIC
-- MAGIC # Plot top dropoff locations in red
-- MAGIC for location in top_dropoff_locations_list:
-- MAGIC     dropoff_location = location['dropoff_location']
-- MAGIC     folium.CircleMarker(
-- MAGIC         location=[dropoff_location[1], dropoff_location[0]],  # (latitude, longitude)
-- MAGIC         radius=5,
-- MAGIC         color='red',
-- MAGIC         fill=False,
-- MAGIC         fill_opacity=0.6,
-- MAGIC         popup=f"Dropoff Count: {location['count']}"
-- MAGIC     ).add_to(nyc_map)
-- MAGIC
-- MAGIC # Display the map
-- MAGIC nyc_map.save("top_pickup_dropoff_locations.html")
-- MAGIC nyc_map

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The busiest pick up and dropoff locations were busy city streets in Manhatten and Long Island, and in front of Newark Liberty International Airport. 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q13 - Fare Amount Distribution Analysis

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC import numpy as np
-- MAGIC
-- MAGIC # Get summary statistics for fare amounts
-- MAGIC df_fares = df.select('fare_amount').toPandas()
-- MAGIC df_fares['fare_amount'] = pd.to_numeric(df_fares['fare_amount'], errors='coerce')
-- MAGIC df_fares = df_fares[df_fares['fare_amount'] > 0]
-- MAGIC
-- MAGIC # Apply log transformation
-- MAGIC df_fares['log_fare_amount'] = np.log(df_fares['fare_amount'])
-- MAGIC
-- MAGIC print(df_fares['fare_amount'].describe().round(2))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Plotting
-- MAGIC plt.figure(figsize=(8, 6))
-- MAGIC plt.boxplot(df_fares['log_fare_amount'], vert=True)
-- MAGIC plt.xlabel("Fare Amount")
-- MAGIC plt.title("Boxplot of Fare Amount with Log-Transformed Scale")
-- MAGIC ticks = plt.gca().get_yticks()
-- MAGIC plt.gca().set_yticklabels([f"{np.exp(tick):.2f}" for tick in ticks])
-- MAGIC
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The fare amounts have a lot of outliers, both very small and very large fares (as high as 3000, and as low as 0.02), however the mean is 11.92. 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q14 - Distance vs Duration Analysis

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Convert `pickup_datetime` and `dropoff_datetime` to timestamp and calculate duration in minutes
-- MAGIC df = df.withColumn("pickup_datetime", F.to_timestamp("tpep_pickup_datetime")) \
-- MAGIC        .withColumn("dropoff_datetime", F.to_timestamp("tpep_dropoff_datetime")) \
-- MAGIC        .withColumn("duration", (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60)
-- MAGIC
-- MAGIC # Create the SQL view with the updated DataFrame
-- MAGIC df.createOrReplaceTempView("tripdata")
-- MAGIC
-- MAGIC # Define distance ranges and calculate average trip duration for each range
-- MAGIC distance_ranges = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 15)]
-- MAGIC
-- MAGIC # Initialize an empty list to store results
-- MAGIC avg_durations = []
-- MAGIC
-- MAGIC # Loop through each distance range and calculate average duration
-- MAGIC for start, end in distance_ranges:
-- MAGIC     query = f"""
-- MAGIC         SELECT '{start}-{end} miles' AS Distance_Range, AVG(duration) AS Average_Duration
-- MAGIC         FROM tripdata
-- MAGIC         WHERE trip_distance >= {start} AND trip_distance < {end}
-- MAGIC     """
-- MAGIC     result = spark.sql(query)
-- MAGIC     avg_durations.append(result)
-- MAGIC
-- MAGIC # Combine all results into a single DataFrame
-- MAGIC avg_durations_df = avg_durations[0]
-- MAGIC for i in range(1, len(avg_durations)):
-- MAGIC     avg_durations_df = avg_durations_df.union(avg_durations[i])
-- MAGIC
-- MAGIC # Show the final average durations DataFrame
-- MAGIC avg_durations_df.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Collect trip distance and duration data
-- MAGIC distance_duration_df = df.select("trip_distance", "duration").toPandas()
-- MAGIC
-- MAGIC # Plotting
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.scatter(distance_duration_df['trip_distance'], distance_duration_df['duration'], alpha=0.5)
-- MAGIC plt.title('Trip Distance vs Duration')
-- MAGIC plt.xlabel('Trip Distance (miles)')
-- MAGIC plt.ylabel('Duration (minutes)')
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The table shows that trip duration increases as trip distance increases.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q15: Daily Trend Analysis

-- COMMAND ----------

-- Count daily trips and visualize the trend
SELECT 
    DATE(tpep_pickup_datetime) AS trip_date, 
    COUNT(*) AS trip_count
FROM tripdata
GROUP BY DATE(tpep_pickup_datetime)
ORDER BY trip_date;


-- COMMAND ----------

-- MAGIC %python
-- MAGIC #Visualize the trend with a line chart
-- MAGIC import pandas as pd
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Assuming 'trip_counts_per_day' is the result of the above SQL query
-- MAGIC trip_counts_per_day = spark.sql("""
-- MAGIC     SELECT DATE(tpep_pickup_datetime) AS trip_date, COUNT(*) AS trip_count
-- MAGIC     FROM tripdata
-- MAGIC     GROUP BY DATE(tpep_pickup_datetime)
-- MAGIC     ORDER BY trip_date
-- MAGIC """).toPandas()
-- MAGIC
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.plot(trip_counts_per_day["trip_date"], trip_counts_per_day["trip_count"], marker="o")
-- MAGIC plt.title("Daily Trip Counts in January 2015")
-- MAGIC plt.xlabel("Date")
-- MAGIC plt.ylabel("Number of Trips")
-- MAGIC plt.xticks(rotation=45)
-- MAGIC plt.grid(True)
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Day over day the trips fluctuate from increasing and decreasing. By looking up the days of the week, we can figure out that at the start of every week, the trips decrease, and increase as it progresses towards Saturday, then dips down again on Sunday/Monday, repeating this pattern at different magnitudes.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q17: Time of Day Impact on Passenger Count

-- COMMAND ----------

-- Group trips by hour and calculate the average passenger count
SELECT 
    HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS hour_of_day,
    AVG(CAST(passenger_count AS INT)) AS avg_passenger_count
FROM tripdata
WHERE CAST(passenger_count AS INT) > 0
GROUP BY HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP))
ORDER BY hour_of_day;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Visualization of average passenger counts by hour
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Assuming 'passenger_count_by_hour' is the result of the above SQL query
-- MAGIC passenger_count_by_hour = spark.sql("""
-- MAGIC     SELECT HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS hour_of_day, 
-- MAGIC     AVG(CAST(passenger_count AS INT)) AS avg_passenger_count 
-- MAGIC     FROM tripdata 
-- MAGIC     WHERE CAST(passenger_count AS INT) > 0 
-- MAGIC     GROUP BY HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) 
-- MAGIC     ORDER BY hour_of_day
-- MAGIC """).toPandas()
-- MAGIC
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.plot(passenger_count_by_hour['hour_of_day'], passenger_count_by_hour['avg_passenger_count'], marker="o")
-- MAGIC plt.title("Average Passenger Count by Hour of the Day")
-- MAGIC plt.xlabel("Hour of the Day")
-- MAGIC plt.ylabel("Average Passenger Count")
-- MAGIC plt.grid(True)
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC From the graph we can see that in the early hours of the morning until 4 am (where passenger count peaks), there is a high passenger count, which then drops towards 6 am (lowest count), then progressively increases throughout the day towards midnight.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Q18: Revenue by Day of the Week Analysis
-- MAGIC

-- COMMAND ----------

-- Calculate total revenue by day of the week
SELECT 
    DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS day_of_week,
    SUM(CAST(fare_amount AS DOUBLE)) AS total_revenue
FROM tripdata
WHERE CAST(fare_amount AS DOUBLE) > 0
GROUP BY DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP))
ORDER BY total_revenue DESC;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Bar chart comparing revenue by day of the week
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Assuming 'revenue_by_day' is the result of the above SQL query
-- MAGIC revenue_by_day = spark.sql("""
-- MAGIC     SELECT DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS day_of_week, 
-- MAGIC     SUM(CAST(fare_amount AS DOUBLE)) AS total_revenue 
-- MAGIC     FROM tripdata 
-- MAGIC     WHERE CAST(fare_amount AS DOUBLE) > 0 
-- MAGIC     GROUP BY DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)) 
-- MAGIC     ORDER BY total_revenue DESC
-- MAGIC """).toPandas()
-- MAGIC
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.bar(revenue_by_day['day_of_week'], revenue_by_day['total_revenue'])
-- MAGIC plt.title("Total Revenue by Day of the Week")
-- MAGIC plt.xlabel("Day of the Week")
-- MAGIC plt.ylabel("Total Revenue ($)")
-- MAGIC plt.xticks(rotation=45)
-- MAGIC plt.grid(True, axis='y')
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC From the table and graph we can see that the revenue is lowest in the early days of the week (Sun to Wed, where people would be the most busy), and is highest near the end, likely due to Friday nights and the weekend being off time for most people.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Final report summarizing the analyses

-- COMMAND ----------

-- MAGIC %md
-- MAGIC This report explores key insights from analyzing NYC taxi data in January 2015. The analysis highlights fare patterns, trip durations, peak times, and popular locations.
-- MAGIC
-- MAGIC **Outliers and Fare Analysis:** We first identified trips with extreme fare values—either very high fares or those with zero or negative values, likely due to rare events or data inconsistencies. Interestingly, we found a minimal correlation between fare and trip distance, suggesting that fares are influenced more by base charges and surcharges than distance alone.
-- MAGIC
-- MAGIC **Trip Duration and Passenger Counts:** Average trip durations were consistent across different passenger groups, hovering around 13-14 minutes. Trips with more passengers, however, tended to have shorter durations, possibly due to route optimizations or ride-sharing trends.
-- MAGIC
-- MAGIC **Distance and Fare Clustering:** As expected, fare amounts generally increased with distance. Short trips under 1 mile averaged around $5.34, while those exceeding 5 miles averaged over $31, illustrating a gradual fare increase relative to distance.
-- MAGIC
-- MAGIC **Trip Frequency by Location:** Popular pickup and drop-off spots included Manhattan, Long Island, and Newark Airport. These areas align with high-density travel points, showing where taxi demand is most concentrated within the city.
-- MAGIC
-- MAGIC **Daily and Hourly Travel Trends:** Fridays and Saturdays had the highest trip counts and revenue, reflecting weekend travel demand. Trips typically peaked around 7 PM on weekdays, matching evening commute times, with lower trip counts seen on early weekday mornings.
-- MAGIC
-- MAGIC **Revenue by Payment and Day:** Card payments were most common, with an average fare around $12.50, slightly higher than cash fares. In terms of weekly revenue, Fridays and Saturdays generated the most, while midweek days showed lower totals.
-- MAGIC
-- MAGIC **Trip Duration and Distance Relationship:** Longer trips took proportionally more time, with short trips under 1 mile averaging 6.5 minutes and longer trips of 10-15 miles taking around 32 minutes. This linear relationship aligns with expected travel times across NYC.
-- MAGIC
-- MAGIC **Popular Travel Times:** Analysis of trip frequency and duration by hour showed peak travel around 5-10 PM, with the highest average passenger counts observed during early morning hours and tapering off around mid-day.
-- MAGIC
-- MAGIC Overall, these insights reflect how NYC’s taxi system adapts to diverse travel demands, varying across time, location, and passenger needs.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
