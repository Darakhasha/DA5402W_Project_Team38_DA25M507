from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.sql.functions import col, from_json, window

# 1. Initialize local Spark Session with Kafka connector dependencies configured
import pyspark
from pyspark.sql import SparkSession

# 1. Dynamically detect your PySpark version and Scala compatibility
spark_ver = pyspark.__version__
scala_ver = "2.13" if spark_ver.startswith("4") else "2.12"
kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_{scala_ver}:{spark_ver}"

print("==========================================")
print(f"🚀 Detected PySpark Version: {spark_ver}")
print(f"📦 Auto-selected Connector: {kafka_package}")
print("==========================================")

# 2. Initialize Spark Session with the perfect matching package
spark = SparkSession.builder \
    .appName("TaxiDemandStreaming") \
    .config("spark.jars.packages", kafka_package) \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Define an explicit schema for parsing incoming JSON payloads [9, 10]
taxi_schema = StructType([
    StructField("PULocationID", StringType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True), # Auto-converts string into Spark Timestamp [9]
    StructField("fare_amount", DoubleType(), True)
])

# 3. Connect to the local Kafka broker and read stream
print("Connecting to Kafka topic 'taxi_demand_stream'...")
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "taxi_demand_stream") \
    .option("startingOffsets", "earliest") \
    .load()

# 4. Decode binary payload into string and parse with our schema [11, 12]
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), taxi_schema).alias("data")) \
    .select("data.*")

# 5. Data Preprocessing & Cleaning [9]
# Filter out anomalies (fares must be strictly positive and realistic) [9]
cleaned_stream = parsed_stream.filter(
    (col("fare_amount") > 0) & (col("fare_amount") < 500)
)

# 6. Real-time Feature Engineering with Time-Window Aggregations [8]
# We implement a 5-minute watermark and group data into 5-minute tumbling windows [13, 14]
windowed_demand = cleaned_stream \
    .withWatermark("tpep_pickup_datetime", "5 minutes") \
    .groupBy(
        window(col("tpep_pickup_datetime"), "5 minutes"),
        col("PULocationID")
    ).count().withColumnRenamed("count", "demand")

# 7. Write the resulting streaming stream to local storage in Parquet format [2]
print("Starting stream processing. Features writing to data/taxi_features_clean.parquet...")
query = windowed_demand.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "data/taxi_features_clean.parquet") \
    .option("checkpointLocation", "data/checkpoints") \
    .start()

query.awaitTermination()