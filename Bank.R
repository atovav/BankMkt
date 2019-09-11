#required libraries
if(!require(RCurl)) install.packages("RCurl", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(fastDummies)) install.packages("fastDummies", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

#load libraries
library(RCurl)
library(tidyverse)
library(caret)
library(gridExtra)
library(randomForest)
library(fastDummies)
library(data.table)
library(knitr)

# Bank Marketing dataset:
# http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
# http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

#get data and remove duration variable
x <- getURL("https://raw.githubusercontent.com/atovav/BankMkt/master/bank-additional-full.csv")
customers <- read.csv(text = x, sep=";")
rm(x)
str(customers)
summary(customers)
customers <- customers[, !(names(customers) %in% c("duration"))]

# Divide into train and test
set.seed(2019, sample.kind="Rounding")
test_index <- createDataPartition(y = customers$y, times = 1, p = 0.2, list = FALSE)
train_set <- customers[-test_index,]
test_set <- customers[test_index,]
rm(test_index,customers)

#see structure
str(train_set)

#plot continous variables vs dependent variable
p1 <- train_set %>% ggplot(aes(y,age)) + geom_boxplot()

p2 <- train_set %>% ggplot(aes(y,campaign)) + geom_boxplot()

p3 <- train_set %>% ggplot(aes(y,pdays)) + geom_boxplot()

p4 <- train_set %>% ggplot(aes(y,previous)) + geom_boxplot()

p5 <- train_set %>% ggplot(aes(y,emp.var.rate)) + geom_boxplot()

p6 <- train_set %>% ggplot(aes(y,cons.price.idx)) + geom_boxplot()

p7 <- train_set %>% ggplot(aes(y,cons.conf.idx)) + geom_boxplot()

p8 <- train_set %>% ggplot(aes(y,euribor3m)) + geom_boxplot()

p9 <- train_set %>% ggplot(aes(y,nr.employed)) + geom_boxplot()

grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, nrow = 2)
rm(p1,p2,p3,p4,p5,p6,p7,p8,p9)
#plot distribution cont var
train_set %>% keep(is.numeric) %>%  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#DIST cont variables
cont <- train_set %>% keep(is.numeric) %>% names()
conset <- train_set[cont]
d <- dist(t(as.matrix(conset)))
heatmap(as.matrix(d), labRow = NA, labCol = NA)

#hcluster
h <- hclust(d)
plot(h, cex = 0.65)
groups <- cutree(h, k = 4)
split(names(groups), groups)
rm(h, cont, d,groups)
# PCA cont variables
pca <- prcomp(as.matrix(conset))
summary(pca)
data.frame(pca$x[,1:3], opted=train_set$y) %>% 
  ggplot(aes(PC1,PC2, fill = opted))+
  geom_point(cex=3, pch=21)
rm(conset,pca)
gc()

#plot discrete variables vs dependent var
p1 <- train_set %>% group_by(job) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(job,Prop)) + geom_col()

p2 <- train_set %>% group_by(marital) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(marital,Prop)) + geom_col()

p3 <- train_set %>% group_by(education) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(education,Prop)) + geom_col()

p4 <- train_set %>% group_by(default) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(default,Prop)) + geom_col()

p5 <- train_set %>% group_by(housing) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(housing,Prop)) + geom_col()

p6 <- train_set %>% group_by(loan) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(loan,Prop)) + geom_col()

p7 <- train_set %>% group_by(contact) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(contact,Prop)) + geom_col()

p8 <- train_set %>% group_by(month) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(month,Prop)) + geom_col()

p9 <- train_set %>% group_by(day_of_week) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(day_of_week,Prop)) + geom_col()

p10 <- train_set %>% group_by(poutcome) %>% summarize(Prop = sum(y=="yes")/n()) %>%
  ggplot(aes(poutcome,Prop)) + geom_col()

grid.arrange(p1,p3,nrow = 2)
grid.arrange(p2, p4, p5, p6, p7, p8, p9, p10, nrow = 2)
rm(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)

#distribution of discrete var
train_set %>% keep(is.factor) %>%  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(stat="count")


#create a validation set from the train set
set.seed(20, sample.kind="Rounding")
val_index <- createDataPartition(y = train_set$y, times = 1, p = 0.15, list = FALSE)
strain_set <- train_set[-val_index,]
val_set <- train_set[val_index,]
valid_set <- na.omit(val_set)
rm(val_index, val_set)

#MOdels
set.seed(10, sample.kind="Rounding")
#GLM
gl_model <- train(y ~ job + education + month + poutcome + pdays + nr.employed, data = strain_set, method = "glm")
preds <- predict(gl_model, valid_set)

results <- data_frame(method = "GLM", 
                      Acc = confusionMatrix(valid_set$y,preds)$overall["Accuracy"], 
                      F1 = F_meas(valid_set$y,preds),
                      Balance = confusionMatrix(valid_set$y,preds)$byClass["Balanced Accuracy"],
                      Specificity = confusionMatrix(valid_set$y,preds)$byClass["Specificity"])
kable(results)
rm(gl_model)

#RF
rf <- randomForest(y ~ job + education + month + poutcome + pdays + nr.employed,
                   data = strain_set, sampsize=1000, ntree=5, importance=TRUE)
print(rf)

rf <- randomForest(y ~ job + education + month + poutcome + pdays + nr.employed,
                   data = strain_set, sampsize=5000, importance=TRUE)
print(rf)
preds <- predict(rf, valid_set)
results <- bind_rows(results,
                          data_frame(method="RF",  
                                     Acc = confusionMatrix(valid_set$y,preds)$overall["Accuracy"], 
                                     F1 = F_meas(valid_set$y,preds),
                                     Balance = confusionMatrix(valid_set$y,preds)$byClass["Balanced Accuracy"],
                                     Specificity = confusionMatrix(valid_set$y,preds)$byClass["Specificity"]))
kable(results)
#RF with all the data
rf <- randomForest(y ~ ., data = strain_set, sampsize=5000, ntree=500, importance=TRUE)
print(rf)
preds <- predict(rf, valid_set)
results <- bind_rows(results,
                     data_frame(method="RF All data",  
                                Acc = confusionMatrix(valid_set$y,preds)$overall["Accuracy"], 
                                F1 = F_meas(valid_set$y,preds),
                                Balance = confusionMatrix(valid_set$y,preds)$byClass["Balanced Accuracy"],
                                Specificity = confusionMatrix(valid_set$y,preds)$byClass["Specificity"]))
kable(results)
imp <- varImp(rf)
imp
#taking features with more than 10
keep <- setDT(imp, keep.rownames = "newname")[]
keep <- keep  %>% filter(no > 10) %>% select(newname)
keep <- as.vector(t(keep))
impset <- strain_set[c("y",keep)]
rm(imp)

#RF with important features 
rf <- randomForest(y ~ ., data = impset, sampsize=5000, importance=TRUE)
print(rf)
predsrf <- predict(rf, valid_set)
results <- bind_rows(results,
                     data_frame(method="RF imp",  
                                Acc = confusionMatrix(valid_set$y,predsrf)$overall["Accuracy"], 
                                F1 = F_meas(valid_set$y,predsrf),
                                Balance = confusionMatrix(valid_set$y,predsrf)$byClass["Balanced Accuracy"],
                                Specificity = confusionMatrix(valid_set$y,predsrf)$byClass["Specificity"]))
kable(results)
#str(impset)

#one hot encoding
impset <- dummy_cols(impset, select_columns = c("job","contact","day_of_week", "poutcome"))
str(impset)
valid_set <- dummy_cols(valid_set, select_columns = c("job","contact","day_of_week", "poutcome"))
drops <- c("job","contact","day_of_week", "poutcome")
impset <- impset[ , !(names(impset) %in% drops)]
impset
colnames(impset)[colnames(impset)=="job_blue-collar"] <- "job_bluecollar"
colnames(impset)[colnames(impset)=="job_self-employed"] <- "job_selfemployed"
colnames(valid_set)[colnames(valid_set)=="job_blue-collar"] <- "job_bluecollar"
colnames(valid_set)[colnames(valid_set)=="job_self-employed"] <- "job_selfemployed"

#rf with one shot var
rf <- randomForest(y ~ ., data = impset, sampsize=5000, importance=TRUE)
print(rf)
predsrf <- predict(rf, valid_set)
results <- bind_rows(results,
                     data_frame(method="RF One hot",  
                                Acc = confusionMatrix(valid_set$y,predsrf)$overall["Accuracy"], 
                                F1 = F_meas(valid_set$y,predsrf),
                                Balance = confusionMatrix(valid_set$y,predsrf)$byClass["Balanced Accuracy"],
                                Specificity = confusionMatrix(valid_set$y,predsrf)$byClass["Specificity"]))
kable(results)

### Start predictions on test set with train set
imp <- varImp(rf)
keep <- setDT(imp, keep.rownames = "newname")[]
keep <- keep %>% select(newname)
keep <- as.vector(t(keep))
train_set <- dummy_cols(train_set, select_columns = c("job","contact","day_of_week", "poutcome"))
test_set <- dummy_cols(test_set, select_columns = c("job","contact","day_of_week", "poutcome"))
colnames(train_set)[colnames(train_set)=="job_blue-collar"] <- "job_bluecollar"
colnames(train_set)[colnames(train_set)=="job_self-employed"] <- "job_selfemployed"
colnames(test_set)[colnames(test_set)=="job_blue-collar"] <- "job_bluecollar"
colnames(test_set)[colnames(test_set)=="job_self-employed"] <- "job_selfemployed"
train_set <- train_set[c("y",keep)]

rf <- randomForest(y ~ ., data = train_set, sampsize=5000)
predsrf <- predict(rf, test_set)
resultsf <- data_frame(method = "Final result", 
                       Acc = confusionMatrix(test_set$y,predsrf)$overall["Accuracy"], 
                       F1 = F_meas(test_set$y,predsrf),
                       Balance = confusionMatrix(test_set$y,predsrf)$byClass["Balanced Accuracy"],
                       Specificity = confusionMatrix(test_set$y,predsrf)$byClass["Specificity"])
kable(resultsf)



