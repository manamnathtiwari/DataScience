from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BookRecommender") \
    .getOrCreate()

# Load Parquet files from HDFS
book_pivot = spark.read.parquet("hdfs://localhost:9000/book_recommender/book_pivot.parquet")
final_rating = spark.read.parquet("hdfs://localhost:9000/book_recommender/final_rating.parquet")
books_name = spark.read.parquet("hdfs://localhost:9000/book_recommender/books_name.parquet")

# Show first 5 rows
print("📚 book_pivot")
book_pivot.show(5)

print("🎯 final_rating")
final_rating.show(5)

print("📖 books_name")
books_name.show(5)

# Optional: Stop Spark session
spark.stop()
