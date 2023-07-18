"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

st.set_page_config(page_title="Home" ,layout="wide", initial_sidebar_state="auto")

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
train_data = pd.read_csv("resources/data/train.csv")
movies = pd.read_csv("resources/data/movies.csv")

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home","Visualise Your Data","Recommender System", "Solution Overview","About App"]


    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------

    if page_selection == "Home":
        st.title('Home')
        st.write('#')
        st.image('pictures/company.jpg',use_column_width=True)

    if page_selection == "Visualise Your Data":
        st.title('Visualise Your Data')
        
        #movies_df = movies.copy()

        #print(movies.head())

        col1, col2 = st.columns((1,1))

        movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['year'] = pd.to_numeric(movies['release_year'])
        movies.drop("release_year", inplace=True, axis=1)

        movies['movie_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

        #print(movies.head())

        movies['genres_split'] = movies['genres'].str.split('|')
        movies.drop("genres", inplace=True, axis=1)

        #print(movies.head())

        movies_df = movies.explode('genres_split')

        #print("EXPLODE")
        #print(movies_df.head())

        #st.header('Movie Genre Visualization')
        col1.subheader('Movie Genre Distribution')

        year_range = (int(movies_df['year'].min()), int(movies_df['year'].max()))
        selected_year = col1.slider('Select Year', min_value=year_range[0], max_value=year_range[1], value=year_range)

        #print("YEAR RANGE:", selected_year[0], "", selected_year[1])

        default_genres = movies_df['genres_split'].unique()
        #selected_genres = st.multiselect('Select Genres', default_genres, default=movies_df['genres_split'].unique())

        placeholder = st.sidebar.empty()
        selected_genres = placeholder.multiselect('Select Genres', default_genres, default=movies_df['genres_split'].unique(), key="1")

        #if st.button('Reset Genres'):
        #    selected_genres = default_genres
            #st.experimental_rerun()

        if st.sidebar.button("Reset Genres"):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Select Genres', default_genres, default=movies_df['genres_split'].unique(), key="2")

        #filtered_df = movies_df[movies_df['year'] == selected_year]

        #out = movies_df["year"].isin(range(selected_year[0], selected_year[1]))
        out_year = movies_df["year"].isin(range(selected_year[0], selected_year[1]))
        out_genre = movies_df['genres_split'].isin(selected_genres)
        filtered_df = movies_df[out_year & out_genre]

        #filtered_df = movies_df[out]

        #print(filtered_df.head())

        genre_counts = filtered_df['genres_split'].value_counts()

        plt.figure(figsize=(8, 4))
        plt.bar(genre_counts.index, genre_counts.values)
        plt.xlabel('Genre')
        plt.ylabel('Number of Movies')
        plt.title(f'Movie Genre Distribution in {selected_year}')
        plt.xticks(rotation=90)
        col1.pyplot(plt)

        #genre_counts = filtered_df.groupby('genres_split').size()

        #fig = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values, labels={'x': 'Genre', 'y': 'Number of Movies'}, title=f'Movie Genre Distribution in {selected_year}')
        #st.plotly_chart(fig)

        col2.subheader('User Ratings')

        train_data['DateTime'] = pd.to_datetime(train_data['timestamp'], unit='s')
        train_data['Year'] = train_data['DateTime'].dt.year

        year_range_train = (int(train_data['Year'].min()), int(train_data['Year'].max()))
        selected_year_train = col2.slider('Select Year', min_value=year_range_train[0], max_value=year_range_train[1], value=year_range_train)

        out_year_train = train_data['Year'].isin(range(selected_year_train[0], selected_year_train[1]))

        filtered_train = train_data[out_year_train]

        print(train_data.head())

        plt.figure(figsize=(8, 4))
        ax = sns.countplot(data=filtered_train, x='rating', palette='viridis')#, order=train_data['rating'].value_counts())

        #total_ratings = train_data.shape[0]
        #total_height = len(train_data)   

        ax.set_yticklabels([tick for tick in ax.get_yticks()])

        for p in ax.patches:
            percentage = int(p.get_height())
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        plt.xlabel('Rating Category')
        plt.ylabel('Count')
        plt.title('Number of Ratings for Each Rating Category')
        plt.xticks(rotation=0)
        col2.pyplot(plt)

        st.subheader('Movies Dataframe')

        movie_count_plot = train_data.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()

        merged_movie_plot = pd.merge(movies, movie_count_plot, on='movieId', how="left")
        merged_movie_plot.drop("title", inplace=True, axis=1)
        #merged_movie_plot.drop("title", inplace=True, axis=1)

        st.dataframe(merged_movie_plot, width=1560, column_config={"movieId": "Movie ID", "year": "Year", "movie_title": "Movie Title", "genres_split": "Genres", "count": "Number of Ratings", "mean":"Average Rating"}, hide_index=True)




    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    
    if page_selection == "About App":
        st.title("BUZZ HIVE Analytics")
        st.image('pictures/our_app.jpg',use_column_width=True)
        st.write("""Watching a movie is fun, but finding the next movie is a stressful experience. You scroll Netflix endlessly, watch trailers, wasting about an hour 
        but you still can't decide what to watch; ring a bell?. OUR Recommendation system got you; it answers the "what to watch next?" question. Say goodbye to
         wasting time searching for what to watch next, and hello to OUR movie recommendations that display only movies relevant to you.""")  
        st.write('#') 
        st.title('Why choose OUR APP ?') 
        st.write(""" - Great User Interface; Unique, Appealing And Easy To Use""")
        st.write(""" - Fast Loading Time and High Performance.""")
        st.write(""" - OUR APP asks the user to select three favourite movies, and then recommends ten movies similar to their favourite movies. """)
        st.write(""" - You choose your preferred recommendation method; content-based or collaborative-based. """)
        
        st.write('#') 
        st.title('OUR TEAM ')
        st.write(""" - Chesley Rogerson""")
        st.write(""" - Xolisile Sibiya""")
        st.write(""" - Pertunia Nhlapo""")
        st.write(""" - Onalenna Borakano""")
        st.write(""" - Kwena Matlala""")
        st.write(""" - Matseke Diale""")
        st.write(""" - Seema Masekwameng""")
        
        

    # OUR solution
    if page_selection == "Solution Overview":

        st.title("Solution Overview")
        st.image('pictures/recommender.png',use_column_width=True)
        st.write(""" After weighing on the differences between the collaborative filtering and content based filtering, the former approach wins. 
	The criteria is not only based on the score but the relevance of the movies.""")




if __name__ == '__main__':
    main()
