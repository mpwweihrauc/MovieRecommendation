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


# We write a function that computes the Residual Mean Squared Error ("typical error")
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Simplest model: We predict a new rating to be the average rating of all movies, which gives us a baseline error value
mu <- mean(edx$rating)
naive_RMSE <- RMSE(edx$rating, mu)
naive_RMSE

# We generate a table to record our approaches and the error they generate
rmse_results <- data_frame(method = "Just the average rating", RMSE = naive_RMSE)
rmse_results

# Are different movies rated differently? We compute the average rating b (as in "bias") for a movie i: b_i

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# We predict movie ratings based on the fact that different movies are rated differently
predicted_ratings <- mu + edx %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  .$b_i

model_1_RMSE <- RMSE(edx$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect Model", RMSE = model_1_RMSE))
rmse_results

# We can see that the RMSE improved when we take into account that different movies are rated differently

# Do different users rate different movies differently? We compute b_u, the user-specific effect
# We can see that a few uers rate movies generally higher/lower than others, while most fall in-between
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")


user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

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




# Rounding my predictions
my_reference <- factor(edx$rating)
my_prediction <- predicted_ratings
my_prediction[my_prediction < 0.5] <- 0.5
my_prediction[my_prediction > 5] <- 5

my_prediction <- round(my_prediction, digits = 1)
my_prediction <- round(my_prediction*2)/2
my_prediction <- factor(my_prediction)

confusionMatrix(my_prediction, my_reference)


