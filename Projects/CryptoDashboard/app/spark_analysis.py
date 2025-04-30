# app/spark_analysis.py

from pyspark.sql.functions import col, avg, lag, stddev, lit
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("CryptoAnalysis").getOrCreate()

def analyze_coin(coin: str, days: int):
    # Simulated HDFS path
    hdfs_path = f"/tmp/{coin}_crypto_data"

    # Load data
    df = spark.read.parquet(hdfs_path)

    # Window for technical indicators
    window_spec = Window.orderBy("timestamp").rowsBetween(-4, 0)  # 5-period SMA

    df = df.withColumn("SMA_5", avg("price").over(window_spec))
    df = df.withColumn("EMA_5", (col("price") * 0.1 + lag("price", 1).over(window_spec) * 0.9))  # Simplified EMA
    df = df.withColumn("Volatility", stddev("price").over(window_spec))
    df = df.withColumn("Return", (col("price") - lag("price", 1).over(window_spec)) / lag("price", 1).over(window_spec))

    # Limit to last N rows for display
    result = df.orderBy(col("timestamp").desc()).limit(10)

    # Return as Pandas for Streamlit and Gemini usage
    return result.toPandas()
