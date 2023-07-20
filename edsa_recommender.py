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
#import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

st.set_page_config(page_title="BUZZflix" ,layout="wide", initial_sidebar_state="auto")

image = Image.open('resources/imgs/BUZZHIVE.jpg')
st.sidebar.image(image)

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
#train_data = pd.read_csv("resources/data/train.csv")
#movies = pd.read_csv("resources/data/movies.csv")

#family_preset = ["Adventure","Animation","Children","Comedy","Fantasy"]

#action_preset = ["Action","Crime","War","Sci-Fi","Western"]
#action_preset = ["Action","Crime","Sci-Fi"]#,"Western"]

#romance_preset = ["Drama","Romance","Comedy"]

#suspense_preset = ["Action","Horror","Thriller"]

@st.cache_data
def movie_page():

    train_data = pd.read_csv("resources/data/train.csv")
    movies = pd.read_csv("resources/data/movies.csv")

    movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['release_year'])
    movies.drop("release_year", inplace=True, axis=1)

    movies['movie_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

    movies['genres_split'] = movies['genres'].str.split('|')
    movies.drop("genres", inplace=True, axis=1)

    movies_explode = movies.explode('genres_split')

    movie_count_plot = train_data.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()

    merged_movie_plot = pd.merge(movies, movie_count_plot, on='movieId', how="left")
    merged_movie_plot.drop("title", inplace=True, axis=1)

    order = ["movieId", "movie_title", "year", "genres_split", "count", "mean"]

    merged_movie_plot = merged_movie_plot.reindex(columns=order)

    return movies_explode, merged_movie_plot

@st.cache_data
def user_page():

    train_df = pd.read_csv("resources/data/train.csv")

    train_df['DateTime'] = pd.to_datetime(train_df['timestamp'], unit='s')
    train_df['Year'] = train_df['DateTime'].dt.year

    #print("train")
    #print(train_df.head(3))

    group_users = train_df.groupby('userId')['rating'].agg(['count', 'mean'])

    group_users.rename(columns={'count': 'Number of Ratings', 'mean': 'average_rating'}, inplace=True)

    #print("group")
    #print(group_users.head(3))

    group_users_sorted = group_users.sort_values(by='Number of Ratings', ascending=False).reset_index()

    top_30_users = group_users_sorted.head(30)

    return train_df, top_30_users

def filter_movies(movie_genres, genre_preset):
#    genre_intersection = set(movie_genres) & set(genre_preset)
    
    if len(movie_genres) < 2:
        min_genres_required = 1
    else:   
        min_genres_required = 2
    
    return sum(genre in movie_genres for genre in genre_preset) >= min_genres_required
    #return len(genre_intersection) >= min_genres_required or len(movie_genres) < min_genres_required

def convert_to_stars(rating):
    
    if pd.notna(rating):
    
        rounded_rating = round(rating * 2) / 2
        num_stars = int(rounded_rating)
        half_star = rounded_rating % 1 != 0
        stars = '⭐' * num_stars
        if half_star:
            stars += '½'
        return stars
    else:
        return np.nan

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Instructions","Movies","Users","Solution Overview","About App"]

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
    
    if page_selection == "Instructions":
        st.title(":question: Overview and Instructions")

        col1, col2, col3 = st.columns((1.8,0.5,1))

        col1.subheader("Overview")
        col1.markdown("Overview description")

        st.markdown("#")

        col1.subheader("Models Investigated")

        col1.markdown("- Content Based Algorithm")
        col1.markdown("Explanation")

        col1.markdown("- Collaborative Based Algorithm")
        col1.markdown("Explanation")   

        col3.info("How to Use", icon="ℹ️")

        # Recommendations
        col3.subheader("Recommendations")
        col3.markdown("- Navigate to the 'Recommender System' page")
        #col3.markdown("- Choose a model from the provided options.")
        #col3.markdown("- Enter a piece of text in the designated input field.")
        #col3.markdown("- Click the 'Classify' button.")
        #col3.markdown("- The selected model will be used to classify the entered text and display the predicted classification.")

        # Movie Insights
        col3.subheader("Movie Insights")
        col3.markdown("- Navigate to the 'Movies' page.")
        #col3.markdown("- Use the tabs to filter the dataframe (e.g. All, Anti, Neutral, Pro, or News).")
        #col3.markdown("- The displayed dataframe will update to show only the rows that match the selected sentiment.")

        # User Insights
        col3.subheader("User Insights")
        col3.markdown("- Navigate to the 'User' page.")
        #col3.markdown("- Enter a search term in the provided search box.")
        #col3.markdown("- Click the 'Search' button.")
        #col3.markdown("- All observations from the training data that contain the entered search term will be displayed.")

    if page_selection == "Movies":
        st.title(':movie_camera: Movie Database')

        movies_df, movie_train = movie_page()

        #st.subheader('Movie Genre Distribution')

        year_range = (int(movies_df['year'].min()), int(movies_df['year'].max()))
        selected_year = st.slider('Select Year', min_value=year_range[0], max_value=year_range[1], value=year_range)

        default_genres = movies_df['genres_split'].unique()

        #print(default_genres)

        family_preset = ["Adventure","Animation","Children","Comedy","Fantasy"]

        action_preset = ["Action","Crime","War","Sci-Fi","Western"]
        #action_preset = ["Action","Crime","Sci-Fi"]#,"Western"]

        romance_preset = ["Drama","Romance", "Comedy"]

        #suspense_preset = ["Action","Horror","Thriller"]
        suspense_preset = ["Crime","Horror","Thriller", "Mystery"]

        placeholder = st.empty()
        selected_genres = placeholder.multiselect('Select Genres', default_genres, default=movies_df['genres_split'].unique(), key="msd")

        filter_genres = default_genres

        #st.markdown(
        #    """
        #    <style>
        #    .sidebar .sidebar-content {
        #        display: flex;
        #        flex-direction: column;
        #        align-items: center;
        #    }
        #    </style>
        #    """,
        #    unsafe_allow_html=True,
        #)

        st.sidebar.markdown("#")

        if st.sidebar.button("Reset Genres"):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Select Genres', default_genres, default=movies_df['genres_split'].unique(), key="msreset")
            filter_genres = default_genres

        st.sidebar.markdown("# Genre Presets")

        if st.sidebar.button("Family"):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Select Genres', default_genres, default=family_preset, key="msfam")
            filter_genres = family_preset
            #print(filter_genres)

        if st.sidebar.button("Action"):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Select Genres', default_genres, default=action_preset, key="msact")
            filter_genres = action_preset

        if st.sidebar.button("Romance"):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Select Genres', default_genres, default=romance_preset, key="msrom")
            filter_genres = romance_preset

        if st.sidebar.button("Suspense"):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Select Genres', default_genres, default=suspense_preset, key="mssus")
            filter_genres = suspense_preset

        out_year = movies_df["year"].isin(range(selected_year[0], selected_year[1]))
        out_genre = movies_df['genres_split'].isin(selected_genres)
        filtered_df = movies_df[out_year & out_genre]

        genre_counts = filtered_df['genres_split'].value_counts()

        plt.figure(figsize=(8, 4))
        ax = sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="husl")

        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=8)

        plt.xlabel('')
        plt.ylabel('Number of Movies')
        plt.title(f'Movie Genre Distribution between {selected_year}')
        plt.xticks(rotation=90)
        st.pyplot(plt)

        col1, col2, col3 = st.columns((2,2,2))

        col1.subheader('Movies Dataframe')

        col3.markdown("")

        filter_df_genres = col3.checkbox('Filter Dataframe')
        
        place_df = st.empty()

        #place_df.dataframe(movie_train, width=1560, column_config={"movieId": "Movie ID", "year": "Year", "movie_title": "Movie Title", "genres_split": "Genres", "count": "Number of Ratings", "mean":"Average Rating"}, hide_index=True)

        movie_train['Average Rating'] = movie_train['mean'].apply(convert_to_stars)
        movie_train.drop("mean", inplace=True, axis=1)

        if filter_df_genres:
            #print(filter_genres)
            #filtered_movies = movie_train[movie_train['genres_split'].apply(lambda x: any(genre in x for genre in filter_genres))]
            #filtered_movies = movie_train[movie_train['genres_split'].apply(filter_movies)]
            filtered_movies = movie_train[movie_train['genres_split'].apply(lambda x: filter_movies(x, filter_genres))]
            #place_df.dataframe(filtered_movies, width=1560, column_config={"movieId": "Movie ID", "year": "Year", "movie_title": "Movie Title", "genres_split": "Genres", "count": "Number of Ratings", "mean":"Average Rating"}, hide_index=True)
            place_df.dataframe(filtered_movies, column_config={"movieId": "Movie ID", "year": "Year", "movie_title": "Movie Title", "genres_split": "Genres", "count": "Number of Ratings"}, hide_index=True, use_container_width=True)        
        else:
            #place_df.dataframe(movie_train, column_config={"movieId": "Movie ID", "year": "Year", "movie_title": "Movie Title", "genres_split": "Genres", "count": "Number of Ratings", "mean":"Average Rating"}, hide_index=True, use_container_width=True)
            place_df.dataframe(movie_train, column_config={"movieId": "Movie ID", "movie_title": "Movie Title", "year": "Release Date", "genres_split": "Genres", "count": "Number of Ratings", "mean":"Average Rating"}, hide_index=True, use_container_width=True)

    if page_selection == "Users":
        
        st.title(":people_holding_hands: User Database")

        st.subheader('User Ratings')

        #train_data['DateTime'] = pd.to_datetime(train_data['timestamp'], unit='s')
        #train_data['Year'] = train_data['DateTime'].dt.year

        train_data, top_30 = user_page()

        year_range_train = (int(train_data['Year'].min()), int(train_data['Year'].max()))
        selected_year_train = st.slider('Select Year', min_value=year_range_train[0], max_value=year_range_train[1], value=year_range_train)

        out_year_train = train_data['Year'].isin(range(selected_year_train[0], selected_year_train[1]))

        filtered_train = train_data[out_year_train]

        #print(train_data.head())

        plt.figure(figsize=(8, 4))
        ax = sns.countplot(data=filtered_train, x='rating', palette='viridis')#, order=train_data['rating'].value_counts())

        #total_ratings = train_data.shape[0]
        #total_height = len(train_data)   

        ax.set_yticklabels([])

        #for p in ax.patches:
        #    percentage = int(p.get_height())
        #    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
        #        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        for label in ax.containers:
            #ax.bar_label(label, fmt="%.f%%")
            ax.bar_label(label, fmt="{:.0f}")

        plt.xlabel('Rating Category')
        plt.ylabel('Count')
        #plt.title('Number of Ratings for Each Rating Category')
        plt.title(f'User Rating Distribution between {selected_year_train}')
        plt.xticks(rotation=0)
        st.pyplot(plt)
        #column_config={"movieId": "Movie ID", "year": "Year", "movie_title": "Movie Title", "genres_split": "Genres", "count": "Number of Ratings"}

        st.subheader("Top 30 Users")

        top_30['Average Rating'] = top_30['average_rating'].apply(convert_to_stars)
        #top_30.drop("average_rating", inplace=True, axis=1)

        st.dataframe(top_30, column_config={"userId": "User ID"}, hide_index=True, use_container_width=True)

        #top_30_sorted = top_30.head(30).sort_values(by='Number of Ratings', ascending=False)

        #plt.figure(figsize=(8, 4))
        #ax2 = sns.barplot(x='userId', y='Number of Ratings', data=top_30_sorted)
        #plt.xticks(rotation=90)
        #st.pyplot(plt)




    
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    if page_selection == "About App":
        st.title(":honeybee: BUZZ HIVE Analytics")
        st.image('resources/imgs/our_app.jpg',use_column_width=True)
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

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
