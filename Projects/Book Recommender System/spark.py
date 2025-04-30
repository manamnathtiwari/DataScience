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
    except Exception as e:
        print(f"Spark data loading failed: {e}")
        spark_success = False
except Exception as e:
    print(f"Spark session creation failed: {e}")
    spark_success = False

# 2. Load scikit-learn model and data (this is our main recommendation system)
try:
    # Verify pickle files exist before loading
    import os
    pkl_files = [
        r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\model.pkl',
        r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\book_pivot.pkl',
        r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\books_name.pkl',
        r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\final_rating.pkl'
    ]
    
    if all(os.path.exists(f) for f in pkl_files):
        model = pickle.load(open(pkl_files[0], 'rb'))
        book_pivot = pickle.load(open(pkl_files[1], 'rb'))
        books_name = pickle.load(open(pkl_files[2], 'rb'))
        final_rating = pickle.load(open(pkl_files[3], 'rb'))
        
        # Validate loaded data
        if (hasattr(model, 'kneighbors') and 
            isinstance(book_pivot, pd.DataFrame) and 
            isinstance(books_name, list) and 
            isinstance(final_rating, pd.DataFrame)):
            
            # If Spark failed to get book list, use scikit-learn's list
            if not spark_success:
                book_list = books_name
                img_lookup = final_rating.set_index('title')['img_url'].to_dict()
            
            sklearn_success = True
        else:
            print("Loaded pickle files don't contain expected data types")
            sklearn_success = False
    else:
        print("Some pickle files are missing")
        sklearn_success = False
except Exception as e:
    print(f"Scikit-learn loading failed: {e}")
    sklearn_success = False

# 3. If both failed, use default data
if not spark_success and not sklearn_success:
    print("Both Spark and scikit-learn failed - using default data")
    book_list = ["The Great Gatsby", "To Kill a Mockingbird", "1984", "Pride and Prejudice", "The Hobbit"]
    img_lookup = {book: "https://via.placeholder.com/150" for book in book_list}

# Recommendation function using scikit-learn
def recommend_books(book_name, n=6):
    try:
        if not sklearn_success:
            print("Scikit-learn not available for recommendations")
            return None, None
            
        if book_name not in book_pivot.index:
            print(f"Book '{book_name}' not found in book_pivot index")
            return None, None
            
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distances, indices = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n+1)
        
        rec_books = []
        poster_urls = []
        
        for i in range(1, len(indices.flatten())):  # Skip the first one (itself)
            rec_book = book_pivot.index[indices.flatten()[i]]
            rec_books.append(rec_book)
            
            # Get the poster URL - handle case where book might not be in final_rating
            poster_url = img_lookup.get(rec_book, "https://via.placeholder.com/150")
            poster_urls.append(poster_url)
        
        return rec_books[:n], poster_urls[:n]
    except Exception as e:
        print(f"Error in recommendation for '{book_name}': {e}")
        return None, None

def get_random_books(n=5):
    try:
        if sklearn_success and hasattr(book_pivot, 'index'):
            random_indices = np.random.choice(len(book_pivot.index), size=n, replace=False)
            books = [book_pivot.index[i] for i in random_indices]
        elif spark_success and 'book_df' in locals():
            books = book_df.select("title").sample(fraction=0.1, seed=42).limit(n).toPandas()['title'].tolist()
        else:
            books = book_list[:n]
        
        posters = [img_lookup.get(book, "https://via.placeholder.com/150") for book in books]
        return books[:n], posters[:n]
    except Exception as e:
        print(f"Error getting random books: {e}")
        books = book_list[:n]
        posters = ["https://via.placeholder.com/150"] * len(books)
        return books, posters

# UI Implementation
selected_book = st.selectbox("Type or Select a book", sorted(book_list))

if st.button('Show Recommendation'):
    recommendations = None
    posters = None
    
    # Try scikit-learn first if available
    if sklearn_success:
        recommendations, posters = recommend_books(selected_book)
        if recommendations is None:
            st.warning("Recommendation system encountered an issue with the selected book")
    
    # If scikit-learn failed or not available, show random books
    if not recommendations:
        st.warning("Showing random books")
        recommendations, posters = get_random_books()
    
    # Display results
    if recommendations:
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(recommendations):
                with col:
                    st.text(recommendations[i])
                    st.image(posters[i], width=150, caption=recommendations[i])
    else:
        st.error("Could not generate any recommendations")

# Cleanup
@atexit.register
def shutdown():
    if spark_success:
        try:
            spark.stop()
        except Exception as e:
            print(f"Error stopping Spark: {e}")