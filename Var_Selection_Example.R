# load the relevant packages
packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","recipes","resample","caret","SuperLearner",
              "data.table","nnls","mvtnorm","ranger","xgboost","splines","Matrix","xtable","pROC","arm",
              "polspline","ROCR","cvAUC", "KernelKnn", "gam","glmnet")
for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package,repos='http://lib.stat.cmu.edu/R/CRAN') 
  }
}

#-----------------------------------------------------------------------------------------------------------------------------------------------
# SIMULATE DATA ----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
set.seed(123)
n=1000
sigma <- abs(matrix(runif(25,0,1), ncol=5))
sigma <- forceSymmetric(sigma)
sigma <- as.matrix(nearPD(sigma)$mat)
x <- rmvnorm(n, mean=c(0,.25,.15,0,.1), sigma=sigma)
modelMat<-model.matrix(as.formula(~ (x[,1]+x[,2]+x[,3]+x[,4]+x[,5])^3))
beta<-runif(ncol(modelMat)-1,0,1)
beta<-c(2,beta) # setting intercept
mu <- 1-plogis(modelMat%*%beta) # true underlying risk of the outcome
y<-rbinom(n,1,mu)

hist(mu);mean(y)

x<-data.frame(x)
D<-data.frame(x,y)
# Specify the number of folds for V-fold cross-validation
folds=5
index<-split(1:1000,1:folds)

# Trying screen.corRank
screen.corRank <- function (Y, X, family, method = "pearson", rank = 2, ...) 
{
  listp <- apply(X, 2, function(x, Y, method) {
    ifelse(var(x) <= 0, 1, cor.test(x, y = Y, method = method)$p.value)
  }, Y = Y, method = method)
  whichVariable <- (rank(listp) <= rank)
  return(whichVariable)
}
# And creating a logical vector indicating which variables to select 
whichVariable <-  screen.corRank(Y=D$y, X=D[,-6])


## split data into 5 groups for 5-fold cross-validation 
## we do this here so that the exact same folds will be used in 
## both the SL fit with the R package, and the hand coded SL
splt<-lapply(1:folds,function(ind) D[index[[ind]],])
# view the first 6 observations in the first [[1]] and second [[2]] folds
head(splt[[1]])
head(splt[[2]])

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Fitting the Superlearner: Original Version ---------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

# NO SCREENING
sl.lib <- list("SL.mean", "SL.glmnet")
SLfitY<-SuperLearner(Y=y,X=x,family="binomial",
                   method="method.AUC",
                   SL.library=sl.lib,
                   cvControl=list(V=folds, validRows=index))
SLfitY

# WITH SCREENING
sl.lib <- list("SL.mean", "SL.glmnet", c("SL.glmnet", "screen.corRank"))
SLfitY_scr<-SuperLearner(Y=y,X=x,family="binomial",
                       method="method.AUC",
                       SL.library=sl.lib,
                       cvControl=list(V=folds, validRows=index))
SLfitY_scr
SLfitY_scr$whichScreen

#this is picking the same variables as the screening function done by hand 

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hand-coding Super Learner -----------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
###########  Fitting individual algorithms on the training set (but not the ii-th validation set) ########### 
m1 <- lapply(1:folds,function(ii)weighted.mean(rbindlist(splt[-ii])$y)) #mean - SL function uses weighted.mean
m2 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-6]), 
                                             as.matrix(do.call(rbind,splt[-ii])[,6]), alpha = 1, family="binomial")) #glmnet
#glmnet with screen.corRank
# subset first to remove the outcome variable (position 6), then to select just those variables with TRUE in whichVariable
m3 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-6][,whichVariable]),
                                             as.matrix(do.call(rbind,splt[-ii])[,6]), alpha = 1, family="binomial"))


########### Predict ########### 
p1 <- lapply(1:folds, function(ii) rep(m1[[ii]], nrow(splt[[ii]])))
p2 <- lapply(1:folds, function(ii) predict(m2[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-6]), s="lambda.min", type="response"))
p3 <- lapply(1:folds, function(ii) predict(m3[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-6][,whichVariable]), s="lambda.min", type="response"))

############  update dataframe 'splt' so that column1 is the observed outcome (y) and subsequent columns contain predictions above ########### 
for(i in 1:folds){
  splt[[i]]<-cbind(splt[[i]][,6], p1[[i]], p2[[i]], p3[[i]])
}

########### View the first 6 observations in the first fold ########### 
head(data.frame(splt[[1]]))

########### CV risk ########### 
risk1<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2], labels=splt[[ii]][,1]))    # CV-risk for mean
risk2 <- lapply(1:folds, function(ii) 1-AUC(predictions=splt[[ii]][,3], labels=splt[[i]][,1]))  # CV-risk for glmnet all
risk3 <- lapply(1:folds, function(ii) 1-AUC(predictions=splt[[ii]][,4], labels=splt[[i]][,1]))  # CV-risk for glmnet screen corRank

a<-rbind(cbind("mean",mean(do.call(rbind, risk1),na.rm=T)),
         cbind("glmnet",mean(do.call(rbind, risk2),na.rm=T)),
         cbind("glmnet_select",mean(do.call(rbind,risk3), na.rm=T)))

########### Predictions for metalearner ########### 
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","AC.mean","AC.glmnet_All","AC.glmnet_screen.corRank")
head(X)

bounds = c(0, Inf)
SL.r<-function(A, y, par){
  A<-as.matrix(A)
  names(par)<-c("AC.mean","AC.glmnet_All","AC.glmnet_screen.corRank")
  predictions <- crossprod(t(A),par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}

init <- rep(1/3, 3)
fit <- optim(par=init, fn=SL.r, A=X[,2:4], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])
fit

alpha<-fit$par/sum(fit$par)
alpha


# Coefficients do NOT agree between SuperLearner and hand-coded
SLfitY_scr
alpha

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------------------------------
#CHECKING PREDICTIONS GOING INTO METALEARNER ---------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
# From SuperLearner
head(SLfitY_scr$Z)
head(X)
# So the predictions are different, why is this? 
SLpredict <- cbind(D$y, SLfitY_scr$Z)
head(SLpredict)

#SuperLearner source code for screen.corP
screen.corP <- function(Y, X, family, obsWeights, id, method = 'pearson',
                        minPvalue = 0.1, minscreen = 2)
{
  listp <- apply(X, 2, function(x, Y, method) {
    ifelse(var(x) <= 0, 1, cor.test(x, y=Y, method = method)$p.value)
  },
  Y = Y, method = method)
  whichVariable <- (listp<= minPvalue)
  if(sum(whichVariable)<minscreen){
    warning('number of variables with p value less than minPvalue is less than minscreen')
    whichVariable[rank(listp)<=minscreen] <- TRUE
  }
  return(whichVariable)
}

screen.corP(Y=D$y, X=D[,-6])

screen.corRank <- function (Y, X, family, method = "pearson", rank = 2, ...) 
{
  listp <- apply(X, 2, function(x, Y, method) {
    ifelse(var(x) <= 0, 1, cor.test(x, y = Y, method = method)$p.value)
  }, Y = Y, method = method)
  whichVariable <- (rank(listp) <= rank)
  return(whichVariable)
}


#SuperLearner source code for screen.corRank
screen.corRank <- function(Y,X,family, method='pearson', rank=2){
  listp <- apply(X,2,function(x,Y,method){
    ifelse(var(x) <= 0, 1, cor.test(x,y=Y, method = method)$p.value)
  }, Y=Y, method=method)
  whichVariable <- (rank(listp)<= rank)
  return(whichVariable)
}
corRank <- screen.corRank(Y=D$y, X=D[,-6])


#SuperLearner source code for screen.randomForest
screen.randomForest <- function(Y, X, family, nVar = 10, ntree = 1000, 
                                mtry = ifelse(family$family == "gaussian", floor(sqrt(ncol(X))),
                                              max(floor(ncol(X)/3), 1)),
                                nodesize = ifelse(family$family == "gaussian",5,1), maxnodes = NULL){
  .SL.require('randomForest')
  if(family$family == "gaussian"){
    rank.rf.fit <- randomForest::randomForest(Y~., data = X, ntree = ntree, mtry = mtry, 
                                              nodesize = nodesize, keep.forest = FALSE,
                                              maxnodes = maxnodes)
  }
  if(family$family == "binomial"){
    rank.rf.fit <- randomForest::randomForest(as.factor(Y)~., data = X,
                                              ntree = ntree, mtry = mtry,
                                              nodesize = nodesize, keep.forest = FALSE,
                                              maxnodes = maxnodes)
  }
  whichVariable <- (rank(-rank.rf.fit$imporance)<=nVar)
  return(whichVariable)
}
screen.randomForest(Y=D$y, X=D[,-6])


