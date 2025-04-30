import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row
import pandas as pd
import atexit

st.header("Book Recommender System using PySpark")

# Initialize Spark with optimized config
spark = SparkSession.builder \
    .appName("BookRecommender") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

# Load data
rating_df = spark.read.parquet(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Projects\Book Recommender System\artificats\final_rating.parquet')
book_df = spark.read.parquet(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Projects\Book Recommender System\artificats\books_name.parquet')

# Prepare indexers
book_indexer = StringIndexer(inputCol="title", outputCol="bookIndex").fit(rating_df)
user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex").fit(rating_df)

# Transform data
rating_indexed = book_indexer.transform(rating_df)
rating_indexed = user_indexer.transform(rating_indexed)

# Cache the data for better performance
rating_indexed.cache()

# Train ALS model with better parameters
als = ALS(
    userCol="userIndex",
    itemCol="bookIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    rank=10,
    maxIter=15,
    regParam=0.1
)
model = als.fit(rating_indexed)

# Create lookup dictionaries
book_lookup = rating_indexed.select("bookIndex", "title").distinct().toPandas().set_index("bookIndex")["title"].to_dict()
title_to_index = rating_indexed.select("title", "bookIndex").distinct().toPandas().set_index("title")["bookIndex"].to_dict()
img_lookup = book_df.select("title", "img_url").toPandas().set_index("title")["img_url"].to_dict()

# Get unique books for dropdown
unique_books = sorted(list(title_to_index.keys()))

def recommend_books(book_title, n=6):
    try:
        if book_title not in title_to_index:
            return [], []
            
        book_idx = title_to_index[book_title]
        
        # Create a dummy user with the selected book
        dummy_user = spark.createDataFrame([Row(userIndex=0, bookIndex=book_idx)])
        
        # Get recommendations (excluding the queried book)
        recs = model.recommendForUserSubset(dummy_user, n+1).collect()
        
        if not recs:
            return [], []
            
        recommended_books = []
        for r in recs[0].recommendations:
            book = book_lookup.get(r.bookIndex)
            if book and book != book_title:  # Skip the queried book
                recommended_books.append(book)
        
        poster_urls = [img_lookup.get(b, "https://via.placeholder.com/150") for b in recommended_books]
        return recommended_books[:n], poster_urls[:n]
        
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return [], []

# UI Implementation
selected_book = st.selectbox("Type or Select a book", unique_books)

if st.button('Show Recommendation'):
    books, posters = recommend_books(selected_book)
    
    if not books:
        st.warning("Showing random books as fallback")
        # Get random books from Spark
        random_books = book_df.select("title").sample(fraction=0.2, seed=42).limit(5).toPandas()["title"].tolist()
        books = random_books
        posters = [img_lookup.get(b, "https://via.placeholder.com/150") for b in books]
    
    # Display results
    cols = st.columns(5)
    for i, (book, poster) in enumerate(zip(books, posters)):
        with cols[i % 5]:
            st.text(book)
            st.image(poster, width=150)

# Cleanup
@atexit.register
def shutdown():
    try:
        spark.stop()
    except:
        pass