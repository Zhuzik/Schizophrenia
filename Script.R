library(ada)
library(caret)
library(e1071)

# Зависимая переменная (1-0)
labels_train = read.csv(file='train_labels.csv',head=TRUE,sep=",")
labels_train$Class = factor(labels_train$Class,labels=c('Healthy.Control','Schizophrenic.Patient'))

# Набор сигналов FNC
FNC_train = read.csv(file='train_FNC.csv',head=TRUE,sep=",")
FNC_test = read.csv(file='test_FNC.csv',head=TRUE,sep=",")

x <- FNC_train[,-1]
y <- labels_train$Class
test.x <- FNC_test[,-1]

# Fit ada 
set.seed(2312)
model_FNC_ada <- ada(x, y, test.x, test.y=NULL, loss=c("exponential","logistic"),
                 type=c("discrete","real","gentle"),iter=50, nu=0.1, bag.frac=0.5,
                 model.coef=TRUE,bag.shift=FALSE,max.iter=20,delta=10^(-10),verbose=TRUE)
p_FNC_ada <- predict(model_FNC_ada, test.x, type="probs")


#Fit a Linear SVM
CV_Folds <- createMultiFolds(y, k = 10, times = 5)

set.seed(2312)
model_FNC_svm <- train(x,y,method="svmLinear",tuneLength=5,
                   trControl=trainControl(method='repeatedCV',index=CV_Folds, classProbs = TRUE))
p_FNC_svm <- predict(model_FNC_svm,test.x,type='prob')

# Набор сигналов SBM
SBM_train = read.csv(file='train_SBM.csv',head=TRUE,sep=",")
SBM_test = read.csv(file='test_SBM.csv',head=TRUE,sep=",")

x <- SBM_train[,-1]
y <- labels_train$Class
test.x <- SBM_test[,-1]

# Fit ada
set.seed(2312)
model_SBM_ada <- ada(x, y, test.x, test.y=NULL, loss=c("exponential","logistic"),
                 type=c("discrete","real","gentle"),iter=50, nu=0.1, bag.frac=0.5,
                 model.coef=TRUE,bag.shift=FALSE,max.iter=20,delta=10^(-10),verbose=TRUE)
p_SBM_ada <- predict(model_SBM_ada), test.x, type="probs")

#Fit a Linear SVM
set.seed(2312)
model_SBM_svm <- train(x,y,method="svmLinear",tuneLength=5,
                   trControl=trainControl(method='repeatedCV',index=CV_Folds, classProbs = TRUE))
p_SBM_svm <- predict(model_SBM_svm,test.x,type='prob')

p_ada <- (p_FNC_ada+p_SBM_ada)/2
p_svm <- (p_FNC_svm+p_SBM_svm)/2
p <- (p_ada+p_ada)/2

result = read.csv(file='result_example.csv',head=TRUE,sep=",")
scores <- p[,2]
result$Probability = scores
write.csv(result,file='result.csv',row.names=FALSE, quote=F)
