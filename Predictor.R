library(lattice)
library(ggplot2)
library(corrplot)
library(caret)
library(randomForest)

wm <- read.csv("C:\\DEP\\Machine Learning\\pml-training.csv", header = TRUE, na.strings = c("NA", ""))
wm_test <- read.csv("C:\\DEP\\Machine Learning\\pml-testing.csv", header = TRUE, na.strings = c("NA", ""))


csums <- colSums(is.na(wm))
csums_log <- (csums == 0)
training_fewer_cols <- wm[, (colSums(is.na(wm)) == 0)]
wm_test <- wm_test[, (colSums(is.na(wm)) == 0)]



del_cols_log <- grepl("X|user_name|timestamp|new_window", colnames(training_fewer_cols))
training_fewer_cols <- training_fewer_cols[, !del_cols_log]
wm_test_final <- wm_test[, !del_cols_log]


inTrain = createDataPartition(y = training_fewer_cols$classe, p = 0.7, list = FALSE)
small_train = training_fewer_cols[inTrain, ]
small_valid = training_fewer_cols[-inTrain, ]


#orMat <- cor(small_train[, -54])
corrplot(cor(small_train[, -54]), order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))

preProc <- preProcess(small_train[, -54], method = "pca", thresh = 0.99)
trainPC <- predict(preProc, small_train[, -54])
valid_testPC <- predict(preProc, small_valid[, -54])

modelFit <- train(small_train$classe ~ ., method = "rf", data = trainPC, trControl = trainControl(method = "cv", number = 4), importance = TRUE)
varImpPlot(modelFit$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, cex = 1, main = "Importance of the Individual Principal Components")

pred_valid_rf <- predict(modelFit, valid_testPC)
confus <- confusionMatrix(small_valid$classe, pred_valid_rf)
confus$table

accur <- postResample(small_valid$classe, pred_valid_rf)
model_accuracy <- accur[[1]]
model_accuracy

out_of_sample_error <- 1 - model_accuracy
out_of_sample_error

testPC <- predict(preProc, wm_test_final[, -54])
pred_final <- predict(modelFit, testPC)
pred_final