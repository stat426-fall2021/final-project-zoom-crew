# Film Ratings

## Stat 426 Final Project

Movies are a phenomenon of entertainment that we enjoy on a daily basis. However, not all movies are made equal. There are certain films that reach new heights, while others fall short of our expectations.

One of the metrics we use to rate movies is known as a Metascore. The Metascore is a weighted average of ratings for a movie, which assign more importance to factors like critic reviews and certain publications. This score is supposed to be an objective rating of a film.

We decided that we would look into other forms of data, such as genre of movie, budget, and non-critic reviews to see if we could predict the Metascore of movies.

### The Data

We are using the "IMDb movies extensive dataset" from kaggle to perform this analysis. This dataset is split into a few CSV files, so we worked to compile the necessary information into a single file for analysis.
We performed the proper data cleaning to make sure that it could be used in a model.

A Correlation plot with the 'cool warm' colors showed that there was a high correlation between normal reviews and the Metascore. There are other positive and negative correlations that are weaker but just as interesting, such as Action, Comedy, and Horror movies perform worse than other genres such as Drama and Biography.

```
data = pd.read_csv('ratings_clean.csv')
corr = data.corr()
corr.syle.background_gradient(cmap='coolwarm')
```
![Corr Plot](Images/CorrPlot.png)
