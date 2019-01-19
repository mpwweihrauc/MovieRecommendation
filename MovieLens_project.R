#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes
# The vtreat package will be used for cross-validation purposes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(vtreat)) install.packages("vtreat", repos = "http://cran.us.r-project.org")

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

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#########
# Begin #
#########

#######################################################################
# Exploratory data analysis, we are getting familiar with our dataset #
#######################################################################

# Each row represents a rating given by a user to a movie.
head(edx) 

# We are dealing with over 9 million movie ratings of ~70000 unique users giving ratings to ~ 10700 different movies.
dim(edx)
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# We take a look at the rating distribution: 4 is the most common rating, followed by 3 and 5.
# 0.5 is the least common rating.
edx %>%
  count(rating) %>%
  arrange(desc(n))

# Histrogram of the rating distribution. Here we can observe that full-star ratings are much more common than half-star ones.
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black", fill = "orange") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution") +
  theme_light()

# Some movies are rated more often than others.
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "orange") + 
  scale_x_log10() + 
  xlab("Number of ratings [log10]") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

# Users differ vastly in how many movies they rated.
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "orange") + 
  scale_x_log10() + 
  xlab("Number of ratings [log10]") +
  ylab("NUmber of users") +
  ggtitle("Number of ratings given by users")


########################################################################################
# Data analysis and modeling approach, with some additional visualizations for clarity #
########################################################################################

# First, we determine the metric to evaluate our model by.
# We write a loss-function that computes the Residual Mean Squared Error (RMSE, or "typical error") as
# our measure of accuracy. The value is the typical error in star rating we would make upon predicting
# a movie rating.

RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

# We begin by building a very simple model: We predict a new movie rating to be the average rating of all
# movies in our training dataset edx.
# This gives us our baseline RMSE to compare future model approaches against.
# We observe that the mean movie rating is a pretty generous > 3.5 stars, quite a bit above "average" (as in 2.5).

mu <- mean(edx$rating)
baseline_RMSE <- RMSE(mu, edx$rating)
baseline_RMSE

# We create a table to record our approaches and the RMSEs they generate.
rmse_results <- data_frame(method = "Simplest model: Average rating", RMSE = baseline_RMSE)
rmse_results

# To improve the model, we utilize the fact that different movies are reated differently.
# We compute the deviation of each movies' mean rating from the total mean of all movies "mu".
# We call the resulting variable "b" (as in "bias") for each movie "i": b_i

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# By plotting our computed b_i, we can see that indeed, there are large differences in average movie ratings
movie_avgs %>% qplot(b_i, geom = "histogram", bins = 10, data = ., color = I("black"), fill = I("orange"))

# We predict movie ratings based on the fact that different movies are rated differently by adding computed "b_i" to "mu".
# If an individual movie is on average rated worse than the average rating of all movies "mu",
# we predict that it will be rated lower than "mu" by "b_i", the difference of the individual movie
# average from the total average.

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_1_RMSE <- RMSE(predicted_ratings, validation$rating)
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
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black", fill = "orange")

# Calculation of the user effect b_u.
user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# We predict ratings taking into account b_i and b_u.
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_RMSE <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Combined Movie & User Effects Model", RMSE = model_2_RMSE))
rmse_results

###
# We take a look at which movies we predict to be the best or worst based on our computed b_i.
# In particular, we notice that some of the best or worst movies we predict were rated by very few users.
###

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

# Top 10 best movies according to our prediction with the largest positive b_i
# (movies rated better than average).
movie_avgs %>% left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# Top 10 worst movies according to our prediction with the largest negative b_i
# (movies rated worse than average).
movie_avgs %>% left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# This might be due to an overall low amount of ratings associated with these movies.
# We take a look at the amount of ratings given to them.
edx %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

edx %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

###
# Indeed, the best and worst movies (largest b_i in either direction) mostly have very few ratings,
# sometimes even just a single one.
# Larger estimates of b_i are likely for movies with very few ratings. The same holds true for the
# user effect b_u, in those cases where users only rated a very small number of movies. 
# We can penalize these by making use of regularization.
# We determine the Lambda that minimizes RMSE. This shrinks the b_i and b_u in case of small number
# of ratings.
# Essentially, by shrinking our estimates when we are rather unsure, we are being more conservative
# in our estimations.
###

# We use a function from the vtreat package to generate k splits of our data for cross-validation.



set.seed(1)
splitPlan <- kWayCrossValidation(nRows = nrow(edx), nSplits = 5, NULL, NULL) # We split our training data into k = 5 different train/test sets for cross-validation

lambdas <- seq(1.5, 3, 0.25) # We define the range of values we test for Lambda. The range has been set as small as possible to reduce computation times.
opt_lambda <- 0 # We initialize an empty vector that takes the results of the for-loop below

# NOTE: This code likely runs for several minutes, please be patient.

for (i in 1:length(splitPlan)){
  
split <- splitPlan[[i]]

rmses <- sapply(lambdas, function(lambda){
  
b_i <- edx[split$train, ] %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lambda), n_i = n()) # Lambda is used to penalize b_i for small n()

# We generate a temporary test-set out of our training data edx
# by using the split from kWayCrossValidation. No information from our true validation set is used.
test_temp <- edx[split$app, ] %>%
  semi_join(edx[split$train, ], by = "movieId") %>%
  semi_join(edx[split$train, ], by = "userId")

# Prediction of ratings with temporary test-set
predicted_ratings <- test_temp %>%
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred

# Calculation of the resulting RMSE with Lambda
return(RMSE(predicted_ratings, test_temp$rating))
})

opt_lambda[i] <- lambdas[which.min(rmses)]

}
opt_lambda # All Lambda values from the k = 5 cross validations
b_i_opt_lambda <- mean(opt_lambda) # We use the mean as our optimal Lambda value

# We calculate the regularized movie effect b_i using the optimised Lambda value.

movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + b_i_opt_lambda), n_i = n()) 


# We plot the regularized estimates against the least squares estimates. Some of the most extreme b_i
# are shrunk by regularization.
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size = sqrt(n))) + 
  geom_point(shape = 1, alpha = 0.5)

# Our new top 10 best and worst movies using regularized values for b_i.
# We successfully removed most movies with only very few ratings from the top.

edx %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

edx %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


###
# Regularization for the user effect b_u.
###


lambdas <- seq(4.5, 5.5, 0.25)
opt_lambda <- 0 # Empty vector that takes the output of the for-loop below


# NOTE: This code likely runs for several minutes, please be patient.
for (i in 1:length(splitPlan)){
  
split <- splitPlan[[i]]

rmses <- sapply(lambdas, function(lambda){
    
b_u <- edx[split$train, ] %>% 
      left_join(movie_reg_avgs, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n() + lambda), n_i = n())
    
test_temp <- edx[split$app, ] %>%
      semi_join(edx[split$train, ], by = "movieId") %>%
      semi_join(edx[split$train, ], by = "userId")
    
predicted_ratings <- test_temp %>%
      left_join(movie_reg_avgs, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      .$pred

# Calculation of the resulting RMSE with Lambda
return(RMSE(predicted_ratings, test_temp$rating))
})
  
opt_lambda[i] <- lambdas[which.min(rmses)]
  
}
opt_lambda
b_u_opt_lambda <- mean(opt_lambda)

# We calculate regularized user effect b_u utilizing the optimized Lambda value (should be ~ 5)
user_reg_avgs <- edx %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n() + b_u_opt_lambda), n_i = n()) 


# We predict ratings based on regularized b_i and b_u.

predicted_ratings <- validation %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_3_RMSE <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized Movie & User Effect",  
                                     RMSE = model_3_RMSE))
rmse_results %>% knitr::kable()


########### 
# Results #
###########

###
# Determining the accuracy of the model
###

# While the predicted ratings have the same mean as the original ratings,
# we observe values below 0.5 and above 5 due to the way we calculated them.
# Furthermore, we predicted numeric values and not categorical ratings from 0.5 to 5.
summary(predicted_ratings)
head(predicted_ratings)

# In order to predict star ratings from 0.5 to 5, we have to round our predictions and substitute
# values above 5 and below 0.5 accordingly.

my_prediction <- predicted_ratings
my_prediction <- round(my_prediction/0.5)*0.5
my_prediction[my_prediction <= 0.5] <- 0.5 # Substitute all values below 0.5 with 0.5
my_prediction[my_prediction >= 5] <- 5 # Substitute all values above 5 with 5

# Accuracy of the predictions in star ratings from 0.5 to 5.
mean(my_prediction == validation$rating)

# RMSE after rounding the predictions to fit the rating scheme from 0.5 to 5.
RMSE(my_prediction, validation$rating)


# Final RMSE value of the predicted ratings without rounding
RMSE(predicted_ratings, validation$rating)

############## 
# Conclusion #
##############