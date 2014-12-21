library(caret)
library(class)
library(randomForest)
library(klaR)

train <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!"))
their_test <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!"))

all_class <- train$classe

train$cvtd_timestamp <- as.POSIXct(train$cvtd_timestamp, 
                                   format = "%d/%m/%Y %H:%M")
their_test$cvtd_timestamp <- as.POSIXct(their_test$cvtd_timestamp, 
                                  format = "%d/%m/%Y %H:%M")

train$Date <- as.Date(train$cvtd_timestamp)
their_test$Date <- as.Date(their_test$cvtd_timestamp)

# Select out time course variables
test_vars <- names(their_test[,!apply(their_test, 2, 
                                      FUN = function(x) any(is.na(x)))])
non_model_cols <- c(1:7, 60, 61)

train <- train[, names(train) %in% c(test_vars, "classe")]

set.seed(634146)
inTrain <- createDataPartition(train$classe, p = 0.75, list = FALSE)
train_data <- train[inTrain,]
test_data <- train[-inTrain,]

train_class <- train_data$classe
test_class <- test_data$classe

train_data <- train_data[,-non_model_cols]
test_data <- test_data[,-non_model_cols]

train_preProcess <- preProcess(train_data, 
                               method = c("YeoJohnson", "knnImpute"))
train_preProcess_pca <- preProcess(train_data, 
                               method = c("YeoJohnson", "knnImpute", "pca"),
                               thresh = 0.95)
# train_preProcess_pca99 <- preProcess(train_data, 
#                                    method = c("YeoJohnson", "knnImpute", "pca"),
#                                    thresh = 0.99)
# train_preProcess_ica <- preProcess(train_data, 
#                                method = c("YeoJohnson", "knnImpute", "ica"),
#                                n.comp = 20)

# Rows with incomplete data appear to be cases where a sensor was
# not recorded. Impute those.

train_pre <- predict(train_preProcess, newdata = train_data)
test_pre <- predict(train_preProcess, newdata = test_data)

train_pre_pca <- predict(train_preProcess_pca, 
                          newdata = train_data)
# train_pre_ica <- predict(train_preProcess_ica,
#                          newdata = train_data)
test_pre_pca <- predict(train_preProcess_pca, newdata = test_data)

# Trying some models

train_pre_rf <- randomForest(x=train_pre, y=train_class)
confusionMatrix(test_class, predict(train_pre_rf, newdata = test_pre))

train_pre_svml <- train(x = train_pre, y = train_class, method = "svmLinear")
