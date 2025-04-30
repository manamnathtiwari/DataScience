import streamlit as st
import pickle
import numpy as np
from pyspark.sql import SparkSession
import pandas as pd
import atexit

st.header("Book Recommender System")

# Initialize variables
spark_success = False
sklearn_success = False
book_list = []
img_lookup = {}

# 1. First try to load book details using Spark (silently)
try:
    spark = SparkSession.builder \
        .appName("BookRecommender") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        # Only load the book data we need for display
        book_df = spark.read.parquet(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Projects\Book Recommender System\artificats\books_name.parquet')
        rating_df = spark.read.parquet(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Projects\Book Recommender System\artificats\final_rating.parquet')
        
        # Get book titles and image URLs
        book_list = book_df.select("title").distinct().toPandas()['title'].tolist()
        img_lookup = rating_df.select("title", "img_url").distinct().toPandas().set_index("title")["img_url"].to_dict()
        
        spark_success = True
    except:
        spark_success = False
except:
    spark_success = False

# 2. Load scikit-learn model and data (this is our main recommendation system)
try:
    model = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\model.pkl','rb'))
    book_pivot = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\book_pivot.pkl','rb'))
    final_rating = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\final_rating.pkl','rb'))
    
    # If Spark failed to get book list, use scikit-learn's list
    if not spark_success:
        book_list = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\books_name.pkl','rb'))
        img_lookup = final_rating.set_index('title')['img_url'].to_dict()
    
    sklearn_success = True
except:
    sklearn_success = False

# 3. If both failed, use default data
if not spark_success and not sklearn_success:
    book_list = ["The Great Gatsby", "To Kill a Mockingbird", "1984", "Pride and Prejudice", "The Hobbit"]
    img_lookup = {book: "https://via.placeholder.com/150" for book in book_list}

# Recommendation function using scikit-learn
def recommend_books(book_name, n=6):
    try:
        if not sklearn_success or book_name not in book_pivot.index:
            return None
            
        book_id = np.where(book_pivot.index == book_name)[0][0]
        _, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n+1)
        
        rec_books = [book_pivot.index[i] for i in suggestion[0] if book_pivot.index[i] != book_name]
        poster_urls = [img_lookup.get(book, "https://via.placeholder.com/150") for book in rec_books]
        
        return rec_books[:n], poster_urls[:n]
    except:
        return None

def get_random_books(n=5):
    try:
        if sklearn_success:
            random_indices = np.random.choice(len(book_pivot.index), size=n, replace=False)
            books = [book_pivot.index[i] for i in random_indices]
        elif spark_success:
            books = book_df.select("title").sample(fraction=0.1, seed=42).limit(n).toPandas()['title'].tolist()
        else:
            books = book_list[:n]
        
        posters = [img_lookup.get(book, "https://via.placeholder.com/150") for book in books]
        return books[:n], posters[:n]
    except:
        books = book_list[:n]
        posters = ["https://via.placeholder.com/150"] * len(books)
        return books, posters

# UI Implementation
selected_book = st.selectbox("Type or Select a book", sorted(book_list))

if st.button('Show Recommendation'):
    recommendations = None
    posters = None
    
    # Always use scikit-learn for recommendations
    if sklearn_success:
        recommendations, posters = recommend_books(selected_book)
    
    # If scikit-learn failed, show random books
    if not recommendations:
        recommendations, posters = get_random_books()
    
    # Display results
    if recommendations:
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(recommendations):
                with col:
                    st.text(recommendations[i])
                    st.image(posters[i] if i < len(posters) else "https://via.placeholder.com/150", width=150)

# Cleanup.
@atexit.register
def shutdown():
    if spark_success:
        try:
            spark.stop()
        except:
            pass