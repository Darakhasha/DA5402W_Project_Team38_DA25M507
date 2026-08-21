import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, hour, dayofweek, sin, cos, expr
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType

spark_ver = pyspark.__version__
scala_ver = "2.13" if spark_ver.startswith("4") else "2.12"
kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_{scala_ver}:{spark_ver}"

spark = SparkSession.builder \
    .appName("TaxiDemandBatch") \
    .config("spark.jars.packages", kafka_package) \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Updated schema matching raw Parquet columns
taxi_schema = StructType([
    StructField("PULocationID", IntegerType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("trip_distance", DoubleType(), True)
])

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

raw_df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", "taxi_demand_stream") \
    .option("startingOffsets", "earliest") \
    .option("endingOffsets", "latest") \
    .load()

parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), taxi_schema).alias("data")) \
    .select("data.*")

# Clean outlier records
cleaned_df = parsed_df.filter(
    (col("fare_amount") > 0) & (col("fare_amount") < 500) &
    (col("trip_distance") > 0) & (col("trip_distance") < 100) &
    col("PULocationID").isNotNull()
)

# Aggregate hourly demand and calculate feature averages required by train.py
windowed_demand = cleaned_df \
    .groupBy(
        window(col("tpep_pickup_datetime"), "1 hour"),
        col("PULocationID")
    ).agg(
        expr("count(1)").alias("demand"),
        expr("avg(fare_amount)").alias("avg_fare_amount"),
        expr("avg(trip_distance)").alias("avg_trip_distance")
    ) \
    .withColumn("hour_of_day", hour(col("window.start"))) \
    .withColumn("day_of_week", dayofweek(col("window.start"))) \
    .withColumn("is_weekend", expr("CASE WHEN day_of_week IN (1, 7) THEN 1 ELSE 0 END")) \
    .withColumn("sin_hour", sin(col("hour_of_day") * (2 * 3.14159 / 24))) \
    .withColumn("cos_hour", cos(col("hour_of_day") * (2 * 3.14159 / 24))) \
    .drop("window")

output_path = "data/processed/taxi_demand_features.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"Writing features to {output_path}...")
windowed_demand.write \
    .mode("overwrite") \
    .format("parquet") \
    .save(output_path)

spark.stop()
print("Spark batch job complete.")