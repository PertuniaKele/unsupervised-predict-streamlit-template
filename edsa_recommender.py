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

    group_users = train_df.groupby('userId')['rating'].agg(['count', 'mean'])

    group_users.rename(columns={'count': 'Number of Ratings', 'mean': 'average_rating'}, inplace=True)

    group_users_sorted = group_users.sort_values(by='Number of Ratings', ascending=False).reset_index()

    top_30_users = group_users_sorted.head(30)

    return train_df, top_30_users

def filter_movies(movie_genres, genre_preset):
    
    if len(movie_genres) < 2:
        min_genres_required = 1
    else:   
        min_genres_required = 2
    
    return sum(genre in movie_genres for genre in genre_preset) >= min_genres_required

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
    
#def filter_top_users(df, selected_rating):
#    filtered_users = df[df['rating'] == selected_rating]
#    top_users = filtered_users.nlargest(5, 'number_of_ratings', keep='all').sort_index()
#    return top_users

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","❓ Instructions","🎥 Movies","🧑‍🤝‍🧑 Users","🐝 About Us"] #,"⭐ Solution Overview"

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
                #try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                #except:
                #    st.error("Oops! Looks like this algorithm does't work.\
                #              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    
    if page_selection == "❓ Instructions":
        st.title(":question: Overview and Instructions")

        tab1, tab2, tab3, tab4 = st.tabs(["📑 Overview", "🏠 Recommender System Instructions", "🎥 Movies Instructions", "🧑‍🤝‍🧑 Users Instructions"])

        #col1, col2, col3 = st.columns((1.8,0.2,1.3))

        with tab1:

            st.subheader("Overview")
            st.markdown("The BUZZflix™️ Movie Recommendation System was developed as an unsupervised learning project. It provides the user with personalised movie recommendations based on three movies they have selected.")
            st.markdown("The user can select between two methods of recommendation algorithms being content-based filtering and collaborative-based filtering which will be detailed further below.")
            st.markdown("#")

            st.subheader("Algorithms Investigated")

            st.markdown("- Content-based Filtering")
            #st.markdown("Content-based algorithms analyse movie features to discern similarities to make appropriate suggestions.")
            #st.markdown("Features can include, but are not limited to, genre, the director and the acting cast for example.")
            st.markdown("At the heart of Content-Based algorithms lies a clever analysis of movie features to uncover meaningful similarities and make personalized movie suggestions tailored just for you. These algorithms go beyond simple genres. They delve into a treasure trove of characteristics, such as the director's vision, the captivating cast, and even the thematic elements that form the essence of each movie.")
            st.markdown("By understanding what makes a movie truly unique, Content-Based filtering ensures that your recommendations align perfectly with your preferences, immersing you in a world of cinematic wonders that captivates the imagination.")
            st.markdown("#")

            #st.markdown("- Collaborative-based Filtering")
            #st.markdown("Collaborative-based algorithms are slightly more intricate as they recommend movies based on similarities between users and/or the movies themselves. Collaborative algorithms can take the form of memory-based or model-based methods.")
            #st.markdown("Memory-based techniques utilise explicit user data, such as ratings, to determine similarities while model-based techniques use algorithms such as Singular Value Decomposition (SVD) to estimate how likely a user would be interested in a movie they have not seen before.")   
            #st.markdown("The collaborative filtering algorithm, used in the BUZZflix™️ application, is a model based approach using a specially trained K-Nearest Neighbours (KNN) model.")
            
            st.markdown("- Collaborative-based Filtering")
            st.markdown("Collaborative-based algorithms are slightly more intricate as they recommend movies based on similarities between users and/or the movies themselves. Collaborative algorithms can take the form of memory-based or model-based methods.")
            st.markdown("Memory-based techniques utilise explicit user data, such as ratings, to determine similarities while model-based techniques use algorithms such as Singular Value Decomposition (SVD) and K-Nearest Neighbours (KNN) to estimate how likely a user would be interested in a movie they have not seen before.")   
            #st.markdown("The collaborative filtering algorithm, used in the BUZZflix™️ application, is a model based approach using a specially trained K-Nearest Neighbours (KNN) model.")
            st.markdown("In the BUZZflix™️ application, we employ a highly sophisticated model-based Collaborative Filtering algorithm, utilizing a specially trained KNN model. This intricate approach takes into account the preferences of similar users, enabling us to recommend movies that align perfectly with your tastes and cinematic inclinations.")

        #col3.info("How to Use", icon="ℹ️")

        with tab2:

            # Recommendations
            #st.subheader("Recommendations")
            st.subheader("")
            st.markdown("- Navigate to the 'Recommender System' page.")
            st.markdown("")
            st.markdown("- Choose a recommendation algorithm from the provided options.")
            st.markdown("")
            st.markdown("- Select a movie that is one of your favourites from each drop-down selectbox (Three movies in total).")
            st.markdown("")
            st.markdown("- Click the 'Recommend' button.")
            st.markdown("")
            st.markdown("- The selected algorithm will be used to generate 10 movie recommendations based on your favourite movies.")

        with tab3:

            # Movie Insights
            #st.subheader("Movie Insights")
            st.subheader("")
            st.markdown("- Navigate to the '🎥 Movies' page.")
            st.markdown("")
            st.markdown("- Use the 'Filter by Year' slider to select a period by which the Genre Distribution graph will be filtered.")
            st.markdown("")
            st.markdown("- Use the multi-selectbox denoted by 'Filter by Genre' to filter the Genre Distribution graph according to the genres selected.")
            st.markdown("")
            st.markdown("- You can also make use of the Genre Presets located in the sidebar on the left to filter the graph according to preconstructed groups of genres.")
            st.markdown("")
            st.markdown("- Below the Genre Distribution bar graph, there is a dataframe that displays all movies present in the database.")
            st.markdown("")
            st.markdown("- If one wishes to filter the dataframe by using the presets, tick the 'Filter Dataframe' checkbox and then select the preset you wish to filter by.")

        with tab4:

            # User Insights
            #st.subheader("User Insights")
            st.subheader("")
            st.markdown("- Navigate to the '🧑‍🤝‍🧑 Users' page.")
            st.markdown("")
            #st.markdown("- Use the 'Select Year Range' slider to filter the User Ratings Distribution graph.")
            st.markdown("- Use the 'Filter by Year' slider to select a period by which the User Rating Distribution graph will be filtered.")
            st.markdown("")
            st.markdown("- Use the 'Show Top 30 Active Users' checkbox to display a dataframe that highlights the top 30 users that have given the most ratings.")
            #st.markdown("- The 'Top 30 Users' dataframe displays the users that have provided the highest number of ratings.")

    if page_selection == "🎥 Movies":
        st.title(':movie_camera: Movies')

        movies_df, movie_train = movie_page()

        st.markdown("#### Filter by Year")
        #st.subheader("Filter by Year")
        year_range = (int(movies_df['year'].min()), int(movies_df['year'].max()))
        selected_year = st.slider('Filter by Year', min_value=year_range[0], max_value=year_range[1], value=year_range, label_visibility="collapsed")

        default_genres = movies_df['genres_split'].unique()

        family_preset = ["Adventure","Animation","Children","Comedy","Fantasy"]

        action_preset = ["Action","Crime","War","Sci-Fi","Western"]

        romance_preset = ["Drama","Romance", "Comedy"]

        suspense_preset = ["Crime","Horror","Thriller", "Mystery"]

        col1, col2, col3 = st.columns((0.8,0.2,3))

        col1.markdown("#### Filter by Genre")
        placeholder = col1.empty()
        selected_genres = placeholder.multiselect('Filter by Genre', default_genres, default=movies_df['genres_split'].unique(), key="msd", label_visibility="collapsed")

        filter_genres = default_genres

        st.sidebar.markdown("#")

        if st.sidebar.button("🌀 Reset Genres", use_container_width=True):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Filter by Genre', default_genres, default=movies_df['genres_split'].unique(), key="msreset", label_visibility="collapsed")
            filter_genres = default_genres

        st.sidebar.markdown("# Genre Presets")

        if st.sidebar.button("👨‍👩‍👦 Family", use_container_width=True):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Filter by Genre', default_genres, default=family_preset, key="msfam", label_visibility="collapsed")
            filter_genres = family_preset
            
        if st.sidebar.button("⚔️ Action", use_container_width=True): #"💥"
            placeholder.empty()
            selected_genres = placeholder.multiselect('Filter by Genre', default_genres, default=action_preset, key="msact", label_visibility="collapsed")
            filter_genres = action_preset

        if st.sidebar.button("💖 Romance", use_container_width=True):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Filter by Genre', default_genres, default=romance_preset, key="msrom", label_visibility="collapsed")
            filter_genres = romance_preset

        if st.sidebar.button("🔪 Suspense", use_container_width=True):
            placeholder.empty()
            selected_genres = placeholder.multiselect('Filter by Genre', default_genres, default=suspense_preset, key="mssus", label_visibility="collapsed")
            filter_genres = suspense_preset

        out_year = movies_df["year"].isin(range(selected_year[0], selected_year[1]))
        out_genre = movies_df['genres_split'].isin(selected_genres)
        filtered_df = movies_df[out_year & out_genre]

        genre_counts = filtered_df['genres_split'].value_counts()

        plt.figure(figsize=(8, 4))
        ax = sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="husl")

        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=8)

        plt.xlabel('', fontsize=9)
        plt.ylabel('Number of Movies', fontsize=9)
        plt.title(f'Movie Genre Distribution between {selected_year}', fontsize=10)
        plt.xticks(rotation=90, fontsize=8)
        col3.pyplot(plt)

        col1, col2, col3 = st.columns((2,2,2))

        col1.subheader('Movies Dataframe')

        col3.markdown("")

        filter_df_genres = col3.checkbox('Filter Dataframe')
        
        place_df = st.empty()

        movie_train['Average Rating'] = movie_train['mean'].apply(convert_to_stars)
        movie_train.drop("mean", inplace=True, axis=1)

        if filter_df_genres:
            filtered_movies = movie_train[movie_train['genres_split'].apply(lambda x: filter_movies(x, filter_genres))]
            place_df.dataframe(filtered_movies, column_config={"movieId": "Movie ID", "year": "Year", "movie_title": "Movie Title", "genres_split": "Genres", "count": "Number of Ratings"}, hide_index=True, use_container_width=True)        
        else:
            place_df.dataframe(movie_train, column_config={"movieId": "Movie ID", "movie_title": "Movie Title", "year": "Release Date", "genres_split": "Genres", "count": "Number of Ratings", "mean":"Average Rating"}, hide_index=True, use_container_width=True)

    if page_selection == "🧑‍🤝‍🧑 Users":
        
        st.title(":people_holding_hands: Users")

        st.subheader('User Ratings')

        #train_data, top_30, group_train = user_page()
        train_data, top_30 = user_page()

        st.markdown("#### Filter by Year")
        year_range_train = (int(train_data['Year'].min()), int(train_data['Year'].max()))
        selected_year_train = st.slider('Select Year', min_value=year_range_train[0], max_value=year_range_train[1], value=year_range_train, label_visibility="collapsed")

        out_year_train = train_data['Year'].isin(range(selected_year_train[0], selected_year_train[1]))

        filtered_train = train_data[out_year_train]

        plt.figure(figsize=(8, 4))
        ax = sns.countplot(data=filtered_train, x='rating', palette="RdYlGn") #palette='viridis')#, order=train_data['rating'].value_counts())

        ax.set_yticklabels([])

        for label in ax.containers:
            #ax.bar_label(label, fmt="%.f%%")
            #ax.bar_label(label, fmt="{:.0f}", fontsize=8)
            ax.bar_label(label, labels=[f'{x:,.0f}' for x in label.datavalues] ,fontsize=8)

        col1, col2, col3 = st.columns((1,4,1))

        plt.xlabel('Rating Category', fontsize=9)
        plt.ylabel('Count', fontsize=9)
        plt.title(f'User Rating Distribution between {selected_year_train}', fontsize=10)
        plt.xticks(rotation=0, fontsize=8)
        col2.pyplot(plt)

        #selected_rating = st.selectbox('Rating Category', train_data["rating"].unique())

        #top_5 = filter_top_users(train_data, selected_rating)

        #st.title('Top 5 Users for Selected Rating Category')
        #st.subheader('Select a Rating Category:')   
        #st.write(f'Selected Rating Category: {selected_rating}')

        #filtered_users_df = filter_top_users(selected_rating)

        #st.dataframe(top_5)

        show_users = st.checkbox("Show Top 30 Active Users")

        if show_users:

            st.subheader("Top 30 Active Users")

            top_30['Average Rating'] = top_30['average_rating'].apply(convert_to_stars)
            top_30.drop("average_rating", inplace=True, axis=1)
            top_30.index += 1
            #top_30.reset_index(drop=True, inplace=True)

            st.dataframe(top_30, column_config={"userId": "User ID"}, use_container_width=True)#, hide_index=True)
    
    #if page_selection == "⭐ Solution Overview":
    #    st.title("Solution Overview")
    #    st.write("Describe your winning approach on this page")

    if page_selection == "🐝 About Us":
        
        st.title(":honeybee: We are BUZZHIVE Analytics!")
        #st.image('resources/imgs/our_app.jpg',use_column_width=True)
        st.markdown('') 

        #col1, col2 = st.columns((1,1))

        st.markdown("Watching a movie is fun, but finding a good movie is a time-consuming experience.")  
        st.markdown("You scroll endless entertainment streaming sites, watch trailers, but you still can't decide what to watch. Ring a bell? :bell:")
        st.markdown("BUZZHIVE Analytics' Recommendation System, BUZZflix™️ has your back!")
        st.markdown("It answers the burning 🔥 question so many of us have, 'What to watch next?'")
        st.markdown("Say goodbye to searching and spend more time watching. Say hello to an on-demand Movie Recommendation System that generates personalised movie recommendations to you and for you.")
        st.markdown('') 

        st.image('resources/imgs/our_app.jpg', width=800)
        st.markdown('')
        #st.markdown('') 
        
        st.header('Why choose BUZZflix™️?')

        st.markdown('Personalization')
        st.markdown('- We prioritize you and your unique tastes. Our Recommender System tailors movie recommendations to match your preferences, ensuring you discover movies that resonate with your interests.')
        st.markdown('Intuitive Interface')
        st.markdown("- We've designed BUZZflix with simplicity and ease of use in mind. From interactive graphs to intuitive sliders, our user-friendly interface makes your exploration a breeze.")
        st.markdown('Data-Driven Insights')
        st.markdown('- Dive into the depths of movie data and uncover intriguing insights. Our graphs and dataframes empower you with knowledge about genres, ratings, and user preferences.')
        st.markdown('Community Spirit')
        st.markdown('- BUZZflix thrives on community engagement. Connect with fellow movie enthusiasts, witness the collective ratings, and contribute to the cinematic conversation.')
        st.markdown('Passionate Team')
        st.markdown('- Behind the scenes, a team of dedicated individuals worked tirelessly to bring BUZZflix to life. Their love for movies and technology fueled the creation of this extraordinary application.')
 
        #st.markdown(" - Sleek user interface and easy to use.")
        #st.markdown(" - Fast loading time and high accuracy.")
        #st.markdown(" - You choose your preferred recommendation method, content-based filtering or collaborative-based filtering.")
        




        #col1, col2, col3 = st.columns((1,1,1))

        st.sidebar.title('BUZZHIVE Team')

        st.sidebar.subheader("Team Lead and Head Data Scientist")
        st.sidebar.markdown("Pertunia Nhlapo - pertuniantuliz@gmail.com")

        st.sidebar.subheader("Vice Lead and Lead Data Engineer")
        st.sidebar.markdown("Chesley Rogerson - ckrogerson@gmail.com")

        st.sidebar.subheader("Data Scientist")
        st.sidebar.markdown("Onalenna Borakano - oborakano@gmail.com")

        st.sidebar.subheader("Data Scientist")
        st.sidebar.markdown("Kwena Matlala - kwenamatlala19@gmail.com")

        st.sidebar.subheader("Data Engineer")
        st.sidebar.markdown("Xolisile Sibiya - xolisbiya203@gmail.com")

        st.sidebar.subheader("Domain Expert")
        st.sidebar.markdown("Seema Masekwameng - khutso.km@gmail.com")

        st.sidebar.subheader("Communications Specialist")
        st.sidebar.markdown("Fridar Diale - fridar.mf@gmail.com")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
