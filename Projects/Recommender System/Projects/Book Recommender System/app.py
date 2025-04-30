import pickle
import streamlit as st 
import numpy as np

st.header("Book Recommender System using Machine Learning")

# Load model and data
model = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\model.pkl','rb'))
book_pivot = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\book_pivot.pkl','rb'))
books_name = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\books_name.pkl','rb'))
final_rating = pickle.load(open(r'C:\Users\Manamnath tiwari\OneDrive\Desktop\DataScience\Recommender System\Projects\Book Recommender System\artificats\final_rating.pkl','rb'))

def fetch_poster(suggestion):
    book_names = []
    ids_index = []
    poster_urls = []
    
    for book_id in suggestion[0]:  # Ensure correct iteration
        book_names.append(book_pivot.index[book_id])  # Fixed indexing

    for name in book_names:
        ids = np.where(final_rating['title'] == name)[0]
        if len(ids) > 0:
            ids_index.append(ids[0])  # Ensure valid index
    
    for ids in ids_index:
        poster_urls.append(final_rating.iloc[ids]['img_url'])  # Append URL
    
    return poster_urls  # Return list of poster URLs

def recommended_books(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]  # Use book_name, not books_name
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    poster_urls = fetch_poster(suggestion)

    for i in suggestion[0]:  # Ensure correct indexing
        book_list.append(book_pivot.index[i])  # Fixed indexing
    
    return book_list, poster_urls  # Return results properly

selected_books = st.selectbox("Type or Select a book", books_name)

if st.button('Show Recommendation'):
    recommendation_books, poster_urls = recommended_books(selected_books)
    
    col_list = st.columns(5)
    
    for i, col in enumerate(col_list):
        if i < len(recommendation_books):
            with col:
                st.text(recommendation_books[i])
                st.image(poster_urls[i])
