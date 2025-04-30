import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
import pandas as pd
import atexit

st.header("Book Recommender System using PySpark")

# Initialize Spark session with proper configurations
spark = SparkSession.builder \
    .appName("BookRecommender") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

# Load Parquet files
rating_df = spark.read.parquet(r'C:\\Users\\Manamnath tiwari\\OneDrive\\Desktop\\DataScience\\Projects\\Book Recommender System\\artificats\\final_rating.parquet')
book_df = spark.read.parquet(r'C:\\Users\\Manamnath tiwari\\OneDrive\\Desktop\\DataScience\\Projects\\Book Recommender System\\artificats\\books_name.parquet')

# Data validation
if rating_df.count() == 0 or book_df.count() == 0:
    st.error("Error: One or more data files are empty. Please check your data files.")
    st.stop()

# Prepare indexers
user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex").fit(rating_df)
book_indexer = StringIndexer(inputCol="title", outputCol="bookIndex").fit(book_df)

rating_indexed = user_indexer.transform(rating_df)
rating_indexed = book_indexer.transform(rating_indexed)

# Cache the indexed data for better performance
rating_indexed.persist()

# Train ALS model
als = ALS(
    userCol="userIndex",
    itemCol="bookIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    implicitPrefs=False  # Explicit ratings
)
model = als.fit(rating_indexed)

# Create lookups
book_index_lookup = rating_indexed.select("bookIndex", "title").distinct().toPandas().set_index("bookIndex")["title"].to_dict()
title_to_index = rating_indexed.select("title", "bookIndex").distinct().toPandas().set_index("title")["bookIndex"].to_dict()
img_lookup = rating_df.select("title", "img_url").distinct().toPandas().set_index("title")["img_url"].to_dict()

# For dropdown selection
unique_books = sorted(list(title_to_index.keys())) if title_to_index else []
selected_book = st.selectbox("Type or Select a book", unique_books) if unique_books else st.empty()

def get_safe_random_sample(df, n=5):
    """Safely get random samples from a DataFrame"""
    try:
        count = df.count()
        if count == 0:
            return []
        
        # Ensure we don't divide by zero and handle cases where n > count
        fraction = min(1.0, max(0.0, float(n)/float(max(1, count))))
        return df.sample(fraction=fraction, seed=42).limit(n).collect()
    except Exception as e:
        st.error(f"Error in random sampling: {str(e)}")
        return []

def recommend_random_books(n=5):
    """Recommend random books with proper error handling"""
    try:
        # Get a safe random sample
        random_books = get_safe_random_sample(book_df, n)
        
        if not random_books:
            raise ValueError("No books available for random sampling")
        
        rec_books = []
        poster_urls = []
        
        for row in random_books:
            book_title = row.title
            rec_books.append(book_title)
            poster_urls.append(img_lookup.get(book_title, "https://via.placeholder.com/150"))
        
        return rec_books[:n], poster_urls[:n]
    except Exception as e:
        st.error(f"Error in random recommendation: {str(e)}")
        # Return some default books if everything fails
        default_books = ["The Great Gatsby", "To Kill a Mockingbird", "1984", 
                        "Pride and Prejudice", "The Hobbit"]
        default_urls = ["https://via.placeholder.com/150"] * len(default_books)
        return default_books, default_urls

def recommend_books_from_book(book_title, n=5):
    """Recommend books based on the selected book"""
    try:
        if not book_title or book_title not in title_to_index:
            return recommend_random_books(n)
            
        book_idx = title_to_index[book_title]
        
        # Create a dataframe with the single book
        from pyspark.sql import Row
        user_df = spark.createDataFrame([Row(userIndex=0, bookIndex=book_idx)])
        
        # Get recommendations
        recommendations = model.recommendForUserSubset(user_df, n)
        
        # Collect results safely
        recs = recommendations.collect()
        
        rec_books = []
        poster_urls = []
        
        if recs and hasattr(recs[0], 'recommendations'):
            for row in recs:
                for rec in row.recommendations:
                    book_title = book_index_lookup.get(rec.bookIndex)
                    if book_title:
                        rec_books.append(book_title)
                        poster_urls.append(img_lookup.get(book_title, "https://via.placeholder.com/150"))
        
        # If no recommendations, fall back to random
        if not rec_books:
            return recommend_random_books(n)
            
        return rec_books[:n], poster_urls[:n]
    except Exception as e:
        st.warning("Recommendation system encountered an error. Showing random books instead.")
        return recommend_random_books(n)

if st.button("Show Recommendation") and unique_books:
    try:
        # Try content-based recommendation first
        recommended_books, poster_urls = recommend_books_from_book(selected_book)
        
        # Display recommendations
        if recommended_books:
            col_list = st.columns(5)
            for i, col in enumerate(col_list):
                if i < len(recommended_books):
                    with col:
                        st.text(recommended_books[i])
                        st.image(poster_urls[i], width=150)
        else:
            st.warning("No recommendations available. Please try again.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
elif not unique_books:
    st.error("No books available in the database. Please check your data files.")

# Register Spark session shutdown at exit
@atexit.register
def shutdown():
    try:
        if 'spark' in globals():
            spark.stop()
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")
