#Importing Data analysis libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

##Getting the data
movie_data= pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.data')
movie_data.head() #made this notebook on Kaggle

movie_data= pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.data',sep='\t')
movie_data.head()

#Since there are no columns mentioned in above data, will add the same by referencing readme file.
column_names= ['user_id','item_id','ratings','timestamp']
movie_data= pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.data',sep='\t', names= column_names)
movie_data.head()

##Getting movie titles
movie_items= pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.item',sep='|', encoding='latin-1')
movie_items.head()

#1st and 2nd column in movie_items dataset represents item id and
#movie title and the remaining data is not so important hence ignoring the same.
cols= ['item_id','movie_title','release date','video release date',
       'IMDb URL','unknown','Action','Adventure','Animation',
       'Children','Comedy','Crime','Documentary','Drama','Fantasy',
       'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

movie_items= pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.item',sep='|', encoding='latin-1',names=cols)
movie_items.head()

movie_items= movie_items[['item_id','movie_title']]
movie_items.head()

##Merging both the tables
movies= pd.merge(movie_data, movie_items, on='item_id')
movies.head()

##Importing Visaulization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib  inline  #TO SHOW PLOTS
sns.set_style('white')

##Exploratory Data Analysis
movies['ratings'].hist(bins=70)
plt.show()

#Calculating the mean ratings for movies
movies.groupby('movie_title')['ratings'].mean()
movies.groupby('movie_title')['ratings'].mean().sort_values(ascending=False).head()

#calculating the count for each movie in ascending order
movies.groupby('movie_title')['ratings'].count().sort_values(ascending=False).head()

#creating a new dataframe with mean ratings for movies
ratings= movies.groupby('movie_title')['ratings'].mean()
ratings

ratings= pd.DataFrame(ratings)
ratings.head()
#Adding a new column to new datafram
ratings['num of ratings']= movies.groupby('movie_title')['ratings'].count()
ratings.head()

#plotting the number of ratings
plt.figure(figsize=(10,6))
ratings['num of ratings'].hist(bins=70)
plt.show()

plt.figure(figsize=(10,4))
ratings['ratings'].hist(bins=70)

#checking the relation between ratings and mean-ratings with scatter plot
sns.jointplot(x='ratings',y='num of ratings',data=ratings, alpha=0.5)

##Recommending Similar Movies

'''Here creating a matrix that has the user ids on one axis and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. hence there will be a lot of NaN values, because most people have not seen most of the movies. '''

moviemat= pd.pivot_table(index='user_id', columns='movie_title',data=movies, values='ratings')
moviemat.head()
#Checking the most rated movies
ratings.sort_values('num of ratings',ascending=False).head(10)

#choosing two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.
#and grabing the user ratings of these two movies
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings
liarliar_user_ratings

#using corrwith() method to get correlations between two pandas series
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

similar_to_starwars
similar_to_liarliar

#cleaning this by removing NaN values and using a DataFrame instead of a series
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

#Sorting the movies by correlation to get the similar movie recommendation as Starwars
corr_starwars.sort_values('Correlation',ascending=False).head(10)


'''But here we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (as it was the most popular movie).

Hence fixing this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier) '''

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()

#Now sorting the movies similar to Starwars which have user rating more than 100
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()