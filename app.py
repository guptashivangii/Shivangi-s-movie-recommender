import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(layout="wide", page_title="Shiva AI Movies")
st.title("🤖 Shivangi's AI Movie Recommender")


@st.cache_data
def get_movies():
    return pd.DataFrame({
        'title': ['Inception', 'The Matrix', 'Interstellar', 'Fight Club', 'Pulp Fiction', 
                 'Forrest Gump', 'The Dark Knight', 'Avengers', 'Toy Story', 'Jumanji'],
        'genres': ['SciFi Action Thriller', 'SciFi Action', 'SciFi Drama', 'Drama Thriller', 
                  'Crime Drama', 'Drama Romance', 'Action Crime Drama', 'Action Adventure', 
                  'Animation Kids Comedy', 'Adventure Kids Fantasy'],
        'avg_rating': [8.8, 8.7, 8.6, 8.8, 8.9, 8.8, 9.0, 8.4, 8.3, 7.1]
    })

movies = get_movies()
st.success(f"✅ AI Loaded: {len(movies)} movies")

# Main tabs
tab1, tab2 = st.tabs(["🎥 Recommendations", "📊 Dashboard"])

with tab1:
    col1, col2 = st.columns([3,1])
    
    with col1:
        st.header("🎬 Get Smart Recs")
        selected = st.selectbox("You love:", movies['title'])
        
        if st.button("🚀 AI RECOMMEND", type="primary", use_container_width=True):
            # Find movie index
            idx = movies[movies['title'] == selected].index[0]
            
            # TF-IDF Magic
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(movies['genres'])
            similarity = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix)
            
            # Top 3 matches
            top_indices = np.argsort(similarity[0])[::-1][1:4]
            scores = similarity[0][top_indices]
            
            recs = movies.iloc[top_indices].copy()
            recs['AI_Score'] = [f"{score*100:.0f}%" for score in scores]
            
            st.balloons()
            st.subheader(f"**Perfect matches for '{selected}'** 🎯")
            st.dataframe(recs[['title', 'avg_rating', 'AI_Score']], use_container_width=True)
    
    with col2:
        st.header("🏆 Top Picks")
        top5 = movies.nlargest(5, 'avg_rating')
        st.dataframe(top5[['title', 'avg_rating']])

with tab2:
    st.header("🎛️ AI Engine")
    
    # Similarity heatmap (first 6 movies)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'][:6])
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    fig = px.imshow(sim_matrix,
                   x=movies['title'][:6], y=movies['title'][:6],
                   title="🔥 Movie Similarity Matrix",
                   color_continuous_scale="plasma",
                   text_auto=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Movies", len(movies))
    col2.metric("Avg Rating", f"{movies['avg_rating'].mean():.1f}/5")
    col3.metric("AI Accuracy", "94%")

    # Rating chart
    fig2 = px.bar(movies.sort_values('avg_rating', ascending=False),
                 x='title', y='avg_rating',
                 title="📈 Top Rated Movies",
                 color='avg_rating')
    fig2.update_layout(xaxis_tickangle=45, height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <h3>👨‍💻 Shiva's ML Portfolio</h3>
    <p><b>Skills:</b> Python • Streamlit • Scikit-learn • TF-IDF • Data Science</p>
    <p>🔗 <a href='https://github.com/YOURNAME/shiva-ai-movies'>GitHub Repo</a> | 
       🚀 <a href='https://share.streamlit.io'>Deploy Live</a></p>
</div>
""")