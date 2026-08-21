import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# 1. Dynamically detect your PySpark version and Scala compatibility
spark_ver = pyspark.__version__
scala_ver = "2.13" if spark_ver.startswith("4") else "2.12"
kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_{scala_ver}:{spark_ver}"

print("==========================================")
print(f"🚀 Detected PySpark Version: {spark_ver}")
print(f"📦 Auto-selected Connector: {kafka_package}")
print("==========================================")

# 2. Initialize Spark Session with matching package
spark = SparkSession.builder \
    .appName("TaxiDemandBatch") \
    .config("spark.jars.packages", kafka_package) \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 3. Define explicit schema for parsing incoming JSON payloads
taxi_schema = StructType([
    StructField("PULocationID", StringType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("fare_amount", DoubleType(), True)
])

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

# 4. Connect to Kafka and read in BATCH mode
# CHANGED: Switched from readStream to read, and added endingOffsets="latest"
print("Connecting to Kafka topic 'taxi_demand_stream' in batch mode...")
raw_df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", "taxi_demand_stream") \
    .option("startingOffsets", "earliest") \
    .option("endingOffsets", "latest") \
    .load()

# 5. Decode binary payload into string and parse schema
parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), taxi_schema).alias("data")) \
    .select("data.*")

# 6. Data Preprocessing & Cleaning
cleaned_df = parsed_df.filter(
    (col("fare_amount") > 0) & (col("fare_amount") < 500)
)

# 7. Time-Window Aggregations
# CHANGED: Removed .withWatermark() because batch processing acts on finite data
windowed_demand = cleaned_df \
    .groupBy(
        window(col("tpep_pickup_datetime"), "5 minutes"),
        col("PULocationID")
    ).count().withColumnRenamed("count", "demand")

# 8. Write Parquet output and exit cleanly
# CHANGED: Switched from writeStream to batch write, and removed awaitTermination()
print("Processing batch features. Writing to data/taxi_features_clean.parquet...")
windowed_demand.write \
    .mode("append") \
    .format("parquet") \
    .save("data/taxi_features_clean.parquet")

spark.stop()
print("Batch job finished successfully.")