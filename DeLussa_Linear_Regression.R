setwd("~/Documents/Stockton Graduate/Machine_Learning/Linear Regression")

#Load readr to open the dataset
library(readr)
library(ggplot2)
TrainData_Group1 <- read_csv("TrainData_Group1.csv")

# Check for NA values in the data
sum(is.na(TrainData_Group1))

# Set the seed to make the randomized data reusable
set.seed(123)

# Make the sample size 80% of the actual dataset and a test variable of the remaining data
sample_data <- sample.int(n = nrow(TrainData_Group1), size = floor(.8 * nrow(TrainData_Group1)), replace = F)
train <- TrainData_Group1[sample_data, ]
test  <- TrainData_Group1[-sample_data, ]

# View each dataset
head(train)
head(test)

# Plot to see what the data looks like
x <- cbind(train$X1, train$X2, train$X3, train$X4, train$X5)
y <- cbind(train$Y,train$Y,train$Y,train$Y,train$Y)
plot(x,y, main = "Initial Dataset" )

# Creates both the training and testing data. Sets y to one column of data and x the rest, then check the column names and dimensions to enseure we have the correct columns
y_train <- as.matrix(train[6])
x_train <- as.matrix(train[1:5])
colnames(x_train)
dim(x_train)
colnames(y_train)
dim(y_train)

y_test <- as.matrix(test[6])
x_test <- as.matrix(test[1:5])
colnames(x_test)
dim(x_test)
colnames(y_test)
dim(y_test)

# Creates the training model function
lin_reg <- function(x,y) {
  # Create a vector of 1s where the length is the same as the dataset, then combine the two to create a matrix
  ones <- rep(1, length(y))
  x <- cbind(ones, x)
  
  # OLS (closed-form solution) Calculates the beta values (coefficients) for the linear model
  # The solve function is used to find the values of the betas/coefficients in the equation as well as the inverse of the matrix. the transpose funciton is used to allow for matrix multiplication to work properly. It first matches the number of rows in x to the number of columns in x then is multiplied by the transpose of x again before finally being mulitplied by y. 
  betas <- solve(t(x) %*% x) %*% t(x) %*% y
  # Rounds to 3 decimal places
  betas <- round(betas, 3)
  
  return(betas)
}

# Predicts Y using the betas found above
y_predict <- function(x, betas) {
  
  # Create a column of 1's for x, then combine it with x
  intercepts <- rep(1, nrow(x))
  x <- cbind(intercepts, x)
  
  # Transpose the betas, then x
  matrix_betas <- t(as.matrix(betas))
  matrix_x <- t(as.matrix(x))
  
  # Get the Y^ value, which is the product of matrix_betas and matrix_x 
  y_prediction <- matrix_betas %*% matrix_x
  
  return(y_prediction)
}


errors <- function(y, y_prediction) {
  # Computes the Residual Sum of Squares, prediciton error, that tells us how far the actual data point is from the prediction, which is what we can not explain using our model
  cat("Error results from my algorithm: ")
  RSS = sum((y - y_prediction)^2)
  cat("\nRSS: ", RSS)
  #Computes the Total Sum of Squares, the actual minus the explained, or a measure of the total variation. This is the difference between the actual points and the mean of all the actual values 
  TSS = sum((y - mean(y_prediction))^2)
  cat("\nTSS: ", TSS)
  #Computes the R-Squared, which tells us the percentage of the response variable variation that is explained by the linear regression model
  R2 = 1 - (RSS / TSS)
  cat("\nR^2: ", R2)
  #Computes the Root Mean Square Error, to see how concentrated the data is, around the line of best fit
  RMSE = sqrt(mean((y_prediction - y)**2))
  cat("\nRMSE: ", RMSE)
  return(list(RSS = RSS, TSS = TSS, R2 = R2, RMSE = RMSE))
  
}

# Uses the linear regression function we made from scratch, to predict the betas using the x and y taining data
Betas <- lin_reg(x_train,y_train)
print(Betas)
# Uses the y_predict function created above to 
Y_Prediciton <- y_predict(x_test, Betas)
print(Y_Prediciton)

plot(x=test$Y, y=Y_Prediciton, pch = "x", main = "Actual Vs. My Predicted", xlab = "Testing Data Y Values", ylab = "Predicted Y Values")

# Runs the error function on the test data, and returns the results of each calculation
error <- errors(test$Y, Y_Prediciton)

# Retruns the result of R's built in linear regression funcion, lm()
R_version_lm <- lm(formula = Y ~ X1 + X2 + X3 + X4 + X5, data = TrainData_Group1)
R_version_lm

# Creates a data frame that lists the betas computed from scratch (in the Y column) and R's betas(in the lm column), allowing for an easy comparison of the results.
comparison <- data.frame(B = Betas, lm = round(R_version_lm$coefficients, 3)) 
comparison
