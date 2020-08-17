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
set.seed(123)
SL.glmnet_new <- create.Learner("SL.glmnet",params=list(nfolds=200))

sl.lib <- list("SL.mean", SL.glmnet_new$names)
SLfitY<-SuperLearner(Y=y,X=x,family="binomial",
                     method="method.AUC",
                     SL.library=sl.lib,
                     cvControl=list(V=folds, validRows=index))
SLfitY

SLfitY$Z[,2]

# WITH SCREENING
set.seed(123)
sl.lib <- list("SL.mean", SL.glmnet_new$names, c(SL.glmnet_new$names, "screen.corRank"))
SLfitY_scr<-SuperLearner(Y=y,X=x,family="binomial",
                         method="method.AUC",
                         SL.library=sl.lib,
                         cvControl=list(V=folds, validRows=index))
SLfitY_scr
SLfitY_scr$whichScreen
SLfitY_scr$Z[,2]
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hand-coding Super Learner -----------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
###########  Fitting individual algorithms on the training set (but not the ii-th validation set) ########### 
## For cv.glmnet, had to change to leave-one-out cross-validation. nfolds = 200 because there are 200 observations in each fold in splt. 
m1 <- lapply(1:folds,function(ii) weighted.mean(rbindlist(splt[-ii])$y)) #mean - SL function uses weighted.mean
set.seed(123)
m2 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-6]), 
                                             do.call(rbind,splt[-ii])[,6], lambda=NULL, 
                                             nlambda = 100, type.measure="deviance", nfolds=200, family="binomial", alpha=1)) #glmnet
#glmnet with screen.corRank
# subset first to remove the outcome variable (position 6), then to select just those variables with TRUE in whichVariable
set.seed(123)
m3 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1+., do.call(rbind,splt[-ii])[,-6][,whichVariable]),
                                             do.call(rbind,splt[-ii])[,6], lambda=NULL,
                                             nlambda = 100, type.measure = "deviance", nfolds=200, family="binomial", alpha=1))


########### Predict ########### 
p1 <- lapply(1:folds, function(ii) rep(m1[[ii]], nrow(splt[[ii]])))
set.seed(123)
p2 <- lapply(1:folds, function(ii) predict(m2[[ii]], newx = model.matrix(~-1 + ., do.call(rbind,splt[ii])[,-6]), s="lambda.min", type="response"))
# Checking that the predictions are the same as from the SL function
cbind(sort(SLfitY$Z[,2]),sort(as.numeric(do.call(rbind,p2))),round(sort(SLfitY$Z[,2])-sort(as.numeric(do.call(rbind,p2))),4))
head(sort(SLfitY$Z[,2]))
head(sort(as.numeric(do.call(rbind,p2))))

set.seed(123)
p3 <- lapply(1:folds, function(ii) predict(m3[[ii]], newx = model.matrix(~-1 + ., do.call(rbind,splt[ii])[,-6][,whichVariable]), s="lambda.min", type="response"))
#Checking that predictions match SL function with variable selection 
cbind(sort(SLfitY_scr$Z[,2]),sort(as.numeric(do.call(rbind,p3))),round(sort(SLfitY_scr$Z[,2])-sort(as.numeric(do.call(rbind,p3))),4))
head(sort(SLfitY_scr$Z[,2]))
head(sort(as.numeric(do.call(rbind,p3))))

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

# Compare 
SLfitY_scr
alpha
a

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------