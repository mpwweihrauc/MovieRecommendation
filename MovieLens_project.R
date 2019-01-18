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







#############
### Begin ###
#############

# Exploratory data analysis, we are getting familiar with our dataset

# Each row represents a rating given by a user to a movie
head(edx) 

# We are dealing with ~70000 unique users giving ratings to ~ 10700 different movies
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# Some movies are rated more often than others
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")




# We write a loss-function that computes the Residual Mean Squared Error ("typical error") as
# our measure of accuracy. The value is the typical error in star rating we would make.
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

# We split our dataset into a training and a testing set. We also make sure not to include movieIds or userIds in the test set, which are not in the training set.
set.seed(1)
test_index <- createDataPartition(y = edx_mod$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
test <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set

test_set <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test_set set back into train set

removed <- anti_join(test, test_set)
train_set <- rbind(train, removed)

# A bit of housekeeping
rm(test, train, removed, test_index)

# Simplest model: We predict a new rating to be the average rating of all movies in our train_seting dataset,
# which gives us a baseline RMSE. We observe that the mean movie rating is a pretty generous > 3.5.
mu <- mean(train_set$rating)
baseline_RMSE <- RMSE(mu, train_set$rating)
baseline_RMSE

# We generate a table to record our approaches and the RMSEs they generate.
rmse_results <- data_frame(method = "Simplest model: Just the average rating", RMSE = baseline_RMSE)
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

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_1_RMSE <- RMSE(predicted_ratings, test_set$rating)
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

# Calculation of the user effect b_u.
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# We predict ratings taking into account b_i and b_u.
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Combined Movie & User Effects Model", RMSE = model_2_RMSE))
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

# This might be due to an overall low amount of ratings associated with these movies.
# We take a look at the amount of ratings.
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

# Indeed, the best and worst movies mostly have very few ratings, sometimes even just a single one.
# Larger estimates of b_i are likely for movies with very few ratings. We can penalize these by making
# use of regularization. We determine the Lambda that minimizes RMSE. This shrinks the b_i and b_u in case of small number of ratings.
# Essentially, by shrinking our estimates when we are rather unsure, we are being more conservative in our estimations.


# We use a function from the vtreat package to generate k splits of our data for cross-validation.

if(!require(vtreat)) install.packages("vtreat", repos = "http://cran.us.r-project.org")

set.seed(1)
splitPlan <- kWayCrossValidation(nRows = nrow(train_set), nSplits = 3, NULL, NULL)
# splitPlan[[1]] # Looking at the structure of the first split

lambdas <- seq(1.5, 3, 0.5)
opt_lambda <- c(1:3) # Empty vector that takes the output of the for-loop below

for (i in 1:3){
split <- splitPlan[[i]]
rmses <- sapply(lambdas, function(lambda){
  
b_i <- train_set[split$train, ] %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lambda), n_i = n())

test_temp <- train_set[split$app, ] %>%
  semi_join(train_set[split$train, ], by = "movieId") %>%
  semi_join(train_set[split$train, ], by = "userId")

predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred

return(RMSE(predicted_ratings, test_temp$rating))
})

opt_lambda[i] <- lambdas[which.min(rmses)]

}
opt_lambda
b_i_opt_lambda <- mean(opt_lambda)

# We calculate the regularized movie effect b_i

movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + b_i_opt_lambda), n_i = n()) 



# Regularization for the user effect b_u
# NOTE: This code likely runs for several minutes, please be patient.
set.seed(1)
splitPlan <- kWayCrossValidation(nRows = nrow(train_set), nSplits = 3, NULL, NULL)
# splitPlan[[1]] # Looking at the structure of the first split

lambdas <- seq(4, 6, 0.5)
opt_lambda <- c(1:3) # Empty vector that takes the output of the for-loop below

for (i in 1:3){
  split <- splitPlan[[i]]
  rmses <- sapply(lambdas, function(lambda){
    
    b_u <- train_set[split$train, ] %>% 
      left_join(movie_reg_avgs, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n() + lambda), n_i = n())
    
    test_temp <- train_set[split$app, ] %>%
      semi_join(train_set[split$train, ], by = "movieId") %>%
      semi_join(train_set[split$train, ], by = "userId")
    
    predicted_ratings <- test_temp %>%
      left_join(movie_reg_avgs, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      .$pred
    
    return(RMSE(predicted_ratings, test_temp$rating))
  })
  
  opt_lambda[i] <- lambdas[which.min(rmses)]
  
}
opt_lambda
b_u_opt_lambda <- mean(opt_lambda)

# We calculate regularized user effect b_u utilizing the optimized Lambda value (~ 5)
user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n() + b_u_opt_lambda), n_i = n()) 


# We plot the regularized estimates against the least squares estimates. Some of the most extreme b_i
# are shrunk by regularization.
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape = 1, alpha = 0.5)

# Our new top 10 best and worst movies using regularized values for b_i
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



# We predict ratings based on regularized b_i and b_u

predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_3_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized Movie & User Effect",  
                                     RMSE = model_3_RMSE))
rmse_results %>% knitr::kable()



edx_mod <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
edx_mod <- edx_mod %>% separate_rows(genres, sep = "\\|")













###
# Genre effect
###

genre_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - b_i - b_u - mu))


predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

model_4_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Movie & User Effect + Genre Effect",  
                                     RMSE = model_4_RMSE))
rmse_results %>% knitr::kable()



# We determine accuracy of our predicted ratings,
# which are calculated by adding the regularized movie bias b_i and the 
# regularized user bias b_u to the average rating mu from the actual ratings.

my_reference <- as.factor(test_set$rating)

my_prediction <- predicted_ratings
my_prediction <- round(my_prediction/0.5)*0.5 # We miss all half-star ratings, but gain accuracy compared to including them.
my_prediction[my_prediction <= 0.5] <- 0.5
my_prediction[my_prediction >= 5] <- 5
# my_prediction <- round(my_prediction/0.5)*0.5 # Rounding for half-star ratings; but gives less accuracy as half-star ratings are less common overall than full-star ratings.
my_prediction <- as.factor(my_prediction)

confusionMatrix(my_prediction, my_reference)


# Where are we making the biggest mistakes?

edx %>% mutate(predictions = my_prediction) %>% filter(rating == 0.5 & predictions == 5 & movieId == 858) %>% knitr::kable()
edx %>% mutate(predictions = my_prediction) %>% filter(userId == 3639) %>% summarize(median_rating = median(rating), n = n()) # This user rated 193 novies with a median rating of 4, but gave "The Godfather" a 0.5. Our model can't predict this behaviour.
edx %>% mutate(predictions = my_prediction) %>% filter(userId == 14269) %>% summarize(median_rating = median(rating), n = n()) # This user rated 222 novies with a median rating of 4.5, but gave "The Godfather" a 0.5. Our model can't predict this behaviour.

###
# Naive bayes approach
###

library(e1071)



NBmodel <- naiveBayes(formula = rating ~ movieId + userId, data = edx, laplace = 0) 

test_subset <- true_score %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

NBpredictions <- predict(NBmodel, newdata = edx, threshold = 0.001)
summary(NBpredictions)



  
predtest  <- ifelse(my_prediction == test_set$rating, my_prediction, NBpredictions)

mean(predtest == test_set$rating)

# Final accuracy, best so far: 44.35%
confusionMatrix(predtest, my_reference)










# Random forest : use combine() to combine several RF objects..
library(randomForest)
set.seed(42)
rf_random <- randomForest(as.factor(rating) ~ movieId + userId, data = train_set[1:1000000, ], ntree = 50)
print(rf_random)
plot(rf_random)


mean(rf_random$predicted == train_set$rating[1:1000000])


install.packages("ranger")
library(ranger)

subset <- edx
ranger_forest <- ranger(formula = as.factor(rating) ~ movieId + userId, data = subset, num.trees = 100)

# We make sure that our test_set doesn't contain any movieIds
# or userIds that don't appear in the subset of the train_set we trained on

test_subset <- true_score %>%
  semi_join(subset, by = "movieId") %>%
  semi_join(subset, by = "userId")

# We predict on the test_subset, look at the confusion matrix and determine our accuracy.
ranger_predict <- predict(ranger_forest, data = test_subset)
ranger_forest$confusion.matrix
mean(ranger_predict$predictions == test_subset$rating)













# We split our dataset into a training and a testing set. We also make sure not to include movieIds or userIds in the test set, which are not in the training set.
set.seed(42)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
test <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set

test_set <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test_set set back into train set

removed <- anti_join(test, test_set)
train_set <- rbind(train, removed)

# A bit of housekeeping
rm(test, train, removed, test_index)


fitControl <- trainControl(method = "repeatedcv", number = 3, repeats = 5)
rf_model <- train(as.factor(rating) ~ movieId + userId, data = train_set, method = "rf",
                  trControl = fitControl, verbose = FALSE)



install.packages("recommenderlab")
install.packages("reshape2")
library(recommenderlab)
library(reshape2)


users <- factor(train_set$userId)
movies <- factor(train_set$movieId)


sparse_matrix <- sparseMatrix(as.numeric(users), as.numeric(movies), dimnames = list(as.character(levels(users)), as.character(levels(movies))), x = train_set$rating)

real_rating_matrix <- new("realRatingMatrix", data = sparse_matrix)

head(as(real_rating_matrix, "data.frame")) # We can turn the realRatingMatrix into a dataframe

rrm_norm <- normalize(real_rating_matrix) # We normalize our rrm
rrm_norm

rec_model = Recommender(real_rating_matrix[1:nrow(real_rating_matrix)],
                  method = "UBCF", param = list(normalize = "Z-score", method = "Cosine", nn = 5))


print(rec_model)
names(getModel(rec_model))
getModel(rec_model)$nn

# Predictions, not on the test set, but this fills the user item matrix so that for every user and movie there's a predicted rating
recom_matrix <- predict(rec_model, real_rating_matrix[1:nrow(real_rating_matrix)], type = "ratings")
recom_matrix



# Get ratings list
rec_list <-as(recom_matrix, "list")
head(summary(rec_list))
ratings <- NULL # Initialize vector for for loop
# For all lines in test file, one by one
for ( u in 1:length(test_set[,"rating"]))
{
  # Read userid and movieid from columns 2 and 3 of test data
  userid <- test_set[u,"userId"]
  movieid <- test_set[u,"movieId"]
  
  # Get as list & then convert to data frame all recommendations for user: userid
  u1 <- as.data.frame(rec_list[[userid]])
  # Create a (second column) column-id in the data-frame u1 and populate it with row-names
  # Remember (or check) that rownames of u1 contain are by movie-ids
  # We use row.names() function
  u1$id <- row.names(u1)
  # Now access movie ratings in column 1 of u1
  x = u1[u1$id == movieid, 1]
  # print(u)
  # print(length(x))
  # If no ratings were found, assign 0. You could also
  #   assign user-average
  if (length(x) == 0)
  {
    ratings[u] <- 0
  }
  else
  {
    ratings[u] <- x
  }
  
}
length(ratings)
tx <- cbind(test_set[, 1], round(ratings))