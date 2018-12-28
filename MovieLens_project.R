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
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
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

# We split the dataset edx into sets for training and testing our algorithm. About 20% are used for testing.
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# We exclude movies and users that are not found in the training set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Simplest model: We predict a new rating to be the average rating of all movies in our training dataset,
# which gives us a baseline RMSE. We observe that the mean movie rating is a pretty generous > 3.5.
mu <- mean(train_set$rating)
baseline_RMSE <- RMSE(test_set$rating, mu)
baseline_RMSE

# We generate a table to record our approaches and the RMSEs they generate.
rmse_results <- data_frame(method = "Simplest model: Average rating", RMSE = baseline_RMSE)
rmse_results

# Are different movies rated differently? We compute the average rating "b" (as in "bias") for a movie "i": b_i

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# By plotting our computed b_i, we can see that indeed, there are large differences in average movie ratings
movie_avgs %>% qplot(b_i, geom = "histogram", bins = 10, data = ., color = I("black"))

# We predict movie ratings based on the fact that different movies are rated differently by adding mu and our computed b_is
# If an individual movie is on average rated worse than the average rating of all movies "mu",
# we predict that it will be rated lower than "mu" by "b_i", the differecnce of the individual movie
# average from the total average

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  .$b_i

model_1_RMSE <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect Model", RMSE = model_1_RMSE))
rmse_results

# We can see that the RMSE improved when we take into account that different movies are rated differently.

# Do users rate different movies differently? We compute b_u, the user-specific effect.
# We can see that some users rate movies generally higher/lower than others,
# while most fall in-between, but also that user rating of movies is generally higher
# than a 2.5 "true average" rating. This was reflected previously in the high mean rating of > 3.5.

# Visualization of mean rating by users with over 100 rated movies.
# Some users are very critical, some are rather generous.
train_set %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Some users rate movies more actively than others. A few users actually rated over a thousand movies, while some rated only a few movies.
train_set %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

# Calculation of the user effect b_u
user_avgs <- train_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# We predict ratings taking into account b_i and b_u
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_RMSE <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie & User Effects Model", RMSE = model_2_RMSE))
rmse_results


# We can make use of regularization to improve our accuracy. We take a look at where we made mistakes when
# taking into account only movie to movie variation.

movie_titles <- train_set %>% 
  select(movieId, title) %>%
  distinct()

# Top 10 best movies according to our prediction with movie to movie bias b_i
movie_avgs %>% left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# Top 10 worst movies according to our prediction with movie to movie bias b_i
movie_avgs %>% left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# This might be due to an overall low amount of ratings associated with these. We take a look at the amount of ratings.
train_set %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

train_set %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


# Indeed, the best and worst movies mostly have very few ratings, sometimes even just a single one
# Larger estimates of b_i are likely for movies with very few ratings. We can penalize these by making
# use of regularization. We determine the Lambda that minimizes RMSE.
lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by = "movieId") %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(test_set$rating, predicted_ratings))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# We calculate regularized movie biases b_i with the optimal Lambda determined above
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambdas[which.min(rmses)]), n_i = n()) 

# We plot the regularlized estimates against the least squares estimates. Some of the most extreme b_i
# are shrunk by regularization.
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape = 1, alpha = 0.5)

# Our new top 10 best and worst movies using regularlized values for b_i
train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# We predict ratings with the regularized movie bias b_i
predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3_rmse <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized Movie Effect",  
                                     RMSE = model_3_rmse))
rmse_results %>% knitr::kable()



# We apply regularization to the estimated user bias b_u
lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)

rmses <- sapply(lambdas, function(l){
  
b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
b_u <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
return(RMSE(test_set$rating, predicted_ratings))
})
qplot(lambdas, rmses)
lambdas[which.min(rmses)]

user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambdas[which.min(rmses)]), n_i = n()) 

# We predict new results with regularization accounted for.
predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_4_rmse <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized Movie & User Effect",  
                                     RMSE = model_4_rmse))
rmse_results %>% knitr::kable()


# Accuracy

my_reference <- as.factor(test_set$rating)

my_prediction <- predicted_ratings
my_prediction[my_prediction <= 0.5] <- 0.5
my_prediction[my_prediction >= 5] <- 5
my_prediction <- ceiling(my_prediction/0.5)*0.5
my_prediction <- as.factor(my_prediction)

confusionMatrix(my_prediction, my_reference)$overall["Accuracy"]






# Matrix factorization. We generate a rating matrix y We then add row and column names and convert them to residuals.

rmatrix <- edx %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

rownames(rmatrix)<- rmatrix[, 1]
rmatrix <- rmatrix[, -1]

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

colnames(rmatrix) <- with(movie_titles, title[match(colnames(rmatrix), movieId)])
rmatrix <- sweep(rmatrix, 2, colMeans(rmatrix, na.rm=TRUE))
rmatrix <- sweep(rmatrix, 1, rowMeans(rmatrix, na.rm=TRUE))
