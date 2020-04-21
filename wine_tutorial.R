library(tidyverse)

if(!require(mlflow)) {
  # devtools::install_github("rstudio/mlflow", subdir = "R/mlflow")
  install.packages("mlflow")
  library(mlflow)
  mlflow::install_mlflow()
}

#### https://github.com/mlflow/mlflow/issues/1009
# mlflow::uninstall_mlflow()
Sys.setenv(PATH = paste0(Sys.getenv("PATH"), ";",
                         "C:\\Users\\migue\\anaconda3\\envs\\r-mlflow-1.7.0\\Scripts", ";",
                         "C:\\Users\\migue\\anaconda3\\Scripts"))
# 1 - Terminal: mlflow server
# 2 - R: mlflow_set_tracking_uri("http://localhost:5000")
Sys.setenv(MLFLOW_VERBOSE=TRUE)
Sys.setenv(MLFLOW_CONDA_HOME = "C:/Users/migue/anaconda3/Scripts/conda.exe")
Sys.setenv(MLFLOW_BIN="C:/Users/migue/anaconda3/envs/r-mlflow-1.7.0/Scripts/mlflow.exe")

# Training the Model ------------------------------------------------------

library(mlflow)
library(glmnet)
library(carrier)

set.seed(40)

# Read the wine-quality csv file
data <- read.csv2(here::here("data", "winequality-red.csv"), dec = ".")

# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- sample(1:nrow(data), 0.75 * nrow(data))
train <- data[sampled, ]
test <- data[-sampled, ]

# The predicted column is "quality" which is a scalar from [3, 9]
train_x <- as.matrix(train[, !(names(train) == "quality")])
test_x <- as.matrix(test[, !(names(train) == "quality")])
train_y <- train[, "quality"]
test_y <- test[, "quality"]

alpha <- mlflow_param("alpha", 0.5, "numeric")
lambda <- mlflow_param("lambda", 0.5, "numeric")

with(mlflow_start_run(), {
  
  model <- glmnet(train_x, train_y, 
                  alpha = alpha, lambda = lambda, 
                  family = "gaussian", standardize = FALSE)
  
  predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
  
  predicted <- predictor(test_x)
  
  rmse <- sqrt(mean((predicted - test_y) ^ 2))
  mae <- mean(abs(predicted - test_y))
  r2 <- as.numeric(cor(predicted, test_y) ^ 2)
  
  message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
  message("  RMSE: ", rmse)
  message("  MAE: ", mae)
  message("  R2: ", r2)
  
  mlflow_log_param("alpha", alpha)
  mlflow_log_param("lambda", lambda)
  mlflow_log_metric("rmse", rmse)
  mlflow_log_metric("r2", r2)
  mlflow_log_metric("mae", mae)
  
  mlflow_log_model(predictor, "model")
})

mlflow_run(uri = "examples/wine", entry_point = "train.R")
mlflow_run(uri = "examples/wine", entry_point = "train.R", 
           parameters = list(alpha = 0.1, lambda = 0.5))


mlflow_snapshot() 
