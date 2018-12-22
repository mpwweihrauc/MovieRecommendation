#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
     semi_join(edx, by = "movieId") %>%
     semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings

validation <- validation %>% select(-rating)

# Ratings will go into the CSV submission file below:

write.csv(validation %>% select(userId, movieId) %>% mutate(rating = NA),
          "submission.csv", na = "", row.names=FALSE)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Begin
# Exploratory data analysis, we are getting familiar with our dataset

# Each row represents a rating given by a user to a movie
head(edx) 

# We are dealing with ~70000 unique users giving ratings to ~ 10700 different movies
edx %>% 
  summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# Some movies are rated more often than others
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")




# We write a loss-function that computes the Residual Mean Squared Error ("typical error") as
# our measure of accuracy. The value is the typical error in star rating we would make
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Simplest model: We predict a new rating to be the average rating of all movies in our dataset,
# which gives us a baseline RMSE. We observe that the mean movie rating is a pretty generous > 3.5.
mu <- mean(edx$rating)
baseline_RMSE <- RMSE(edx$rating, mu)
baseline_RMSE

# We generate a table to record our approaches and the RMSEs they generate.
rmse_results <- data_frame(method = "Simplest model: Average rating", RMSE = baseline_RMSE)
rmse_results

# Are different movies rated differently? We compute the average rating "b" (as in "bias") for a movie "i": b_i

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# By plotting our computed b_i, we can see that indeed, there are large differences in average movie ratings
movie_avgs %>% qplot(b_i, geom = "histogram", bins = 10, data = ., color = I("black"))

# We predict movie ratings based on the fact that different movies are rated differently by adding mu and our computed b_is
# If an individual movie is on average rated worse than the average rating of all movies "mu",
# we predict that it will be rated lower than "mu" by "b_i", the differecnce of the individual movie
# average from the total average

predicted_ratings <- mu + edx %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  .$b_i

model_1_RMSE <- RMSE(edx$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect Model", RMSE = model_1_RMSE))
rmse_results

# We can see that the RMSE improved when we take into account that different movies are rated differently.

# Do users rate different movies differently? We compute b_u, the user-specific effect.
# We can see that some users rate movies generally higher/lower than others,
# while most fall in-between, but also that user rating of movies is generally higher
# than a 2.5 "true average" rating. This was reflected previously in the high mean rating of > 3.5.

# Visualization of mean rating by users with over 100 rated movies.
# Some users are very critical, some are rather generous.
edx %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Calculation of the user effect b_u, taking into account b_i
user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# We predict ratings taking into account b_i and b_u
predicted_ratings <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_RMSE <- RMSE(edx$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User Effects Model", RMSE = model_2_RMSE))
rmse_results

# We can see that by taking into account both the movie and the user effect, we can further improve our RMSE


# Matrix factorization. We generate a rating matrix y while removing some less-important entries. We then add row and column names and convert them to residuals.
train_small <- edx %>% 
  group_by(movieId) %>%
  filter(n() >= 50) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

rownames(y)<- y[,1]
y <- y[,-1]

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))