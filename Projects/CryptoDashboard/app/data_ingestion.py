# app/data_ingestion.py

import requests
import json
import redis
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from datetime import datetime
from config import CRYPTO_API_URL, REDIS_HOST, REDIS_PORT

# Spark Session
spark = SparkSession.builder \
    .appName("CryptoDataIngestion") \
    .getOrCreate()

# Redis Connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

def fetch_crypto_data(coin: str, days: int):
    url = f"{CRYPTO_API_URL}/{coin}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from API")

    data = response.json()
    price_data = data.get("prices", [])  # [[timestamp, price], ...]

    records = [{
        "coin": coin,
        "timestamp": datetime.fromtimestamp(ts / 1000),
        "price": price
    } for ts, price in price_data]

    return records

def fetch_and_store_crypto_data(coin: str, days: int):
    records = fetch_crypto_data(coin, days)

    # Save to Redis (latest only)
    latest = records[-1]
    redis_key = f"crypto:{coin}:latest"
    redis_client.set(redis_key, json.dumps(latest))

    # Convert to DataFrame for PySpark
    schema = StructType([
        StructField("coin", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("price", DoubleType(), True)
    ])

    df = spark.createDataFrame(records, schema)

    # Simulated HDFS path (can replace with real HDFS path later)
    hdfs_path = f"/tmp/{coin}_crypto_data"
    df.write.mode("overwrite").parquet(hdfs_path)

    print(f"[✔] Data for {coin} written to Redis and HDFS (simulated at {hdfs_path})")
