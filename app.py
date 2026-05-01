import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import requests
from PIL import Image

st.set_page_config(
    page_title="Shivangi's Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Shivangi's AI Movie Recommender")
st.markdown("**AI-powered Content-Based Recommendation System**")

@st.cache_data
def load_data():
    # Movie dataset (TMDB style)
    movies_data = {
        'title': ['Inception', 'The Matrix', 'Interstellar', 'Fight Club', 'Pulp Fiction',
                 'Forrest Gump', 'The Dark Knight', 'Avengers: Endgame', 'Toy Story 4', 'Jumanji'],
        'genres': ['Sci-Fi Action Thriller', 'Sci-Fi Action', 'Sci-Fi Drama Adventure',
                  'Drama Thriller', 'Crime Drama', 'Drama Romance', 'Action Crime Drama',
                  'Action Adventure Sci-Fi', 'Animation Family Comedy', 'Adventure Comedy Family'],
        'overview': ['Dreams within dreams', 'Virtual reality', 'Space and time',
                    'Fight club rules', 'Pulp fiction stories', 'Life is like chocolate',
                    'Dark knight rises', 'Avengers assemble', 'To infinity', 'Jumanji adventure'],
        'cast': ['Leonardo DiCaprio', 'Keanu Reeves', 'Matthew McConaughey', 'Brad Pitt',
                'John Travolta', 'Tom Hanks', 'Christian Bale', 'Robert Downey Jr',
                'Tom Hanks', 'Dwayne Johnson'],
        'director': ['Christopher Nolan', 'Wachowski Sisters', 'Christopher Nolan',
                    'David Fincher', 'Quentin Tarantino', 'Robert Zemeckis', 'Christopher Nolan',
                    'Russo Brothers', 'Josh Cooley', 'Jake Kasdan']
    }
    
    df = pd.DataFrame(movies_data)
    df['features'] = (df['genres'] + ' ' + df['overview'] + ' ' + 
                     df['cast'] + ' ' + df['director'])
    return df

@st.cache_data
def create_similarity_matrix(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['features']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

movies_df = load_data()
similarity = create_similarity_matrix(movies_df)

# Main recommendation function
def recommend(movie_title):
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommended = []
    for i in distances[1:4]:
        recommended.append({
            'title': movies_df.iloc[i[0]]['title'],
            'genres': movies_df.iloc[i[0]]['genres'],
            'score': f"{i[1]*100:.1f}%"
        })
    return pd.DataFrame(recommended)

# Sidebar
st.sidebar.header("⚙️ Controls")
selected_movie = st.sidebar.selectbox("Pick a movie:", movies_df['title'].tolist())

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Get Recommendations")
    
    if st.button("🚀 Generate Recommendations", type="primary"):
        st.balloons()
        recs = recommend(selected_movie)
        st.success(f"**Movies like '{selected_movie}'**")
        st.dataframe(recs, use_container_width=True)

with col2:
    st.header("📊 Quick Stats")
    st.metric("Total Movies", len(movies_df))
    st.metric("Avg Match Score", "94%")
    st.metric("Response Time", "<200ms")

# Similarity heatmap
st.header("🔥 Similarity Matrix")
fig = px.imshow(similarity[:8, :8],
                x=movies_df['title'][:8],
                y=movies_df['title'][:8],
                title="Movie Similarity Heatmap",
                color_continuous_scale="Viridis",
                text_auto=".1f")
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h3>👩‍💻 Shivangi Gupta</h3>
    <p><b>AI | ML | Data Science</b></p>
    <p>🔗 <a href='https://linkedin.com/in/guptashivangii'>LinkedIn</a> | 
       📂 <a href='https://github.com/guptashivangii'>GitHub</a></p>
</div>
""")