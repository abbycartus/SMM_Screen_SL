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
# A. SIMULATE DATA
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Simulate data set with 1000 observations
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

# Specify number of folds and create index for cross-validation
folds=5
index<-split(1:1000,1:folds)

# Split the data into 5 groups for 5-fold cross-validation
splt<-lapply(1:folds,function(ind) D[index[[ind]],])
# view the first 6 observations in the first [[1]] and second [[2]] folds
head(splt[[1]])
head(splt[[2]])

# screen.corRank from SuperLearner package
screen.corRank <- function (Y, X, family, method = "pearson", rank = 2, ...) 
{
  listp <- apply(X, 2, function(x, Y, method) {
    ifelse(var(x) <= 0, 1, cor.test(x, y = Y, method = method)$p.value)
  }, Y = Y, method = method)
  whichVariable <- (rank(listp) <= rank)
  return(whichVariable)
}



#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------
# B. FITTING SUPERLEARNER USING THE PACKAGE
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Once without any variable screening
set.seed(123)
SL.glmnet_new <- create.Learner("SL.glmnet",params=list(nfolds=200))

sl.lib <- list("SL.mean", SL.glmnet_new$names)
SLfitY<-SuperLearner(Y=y,X=x,family="binomial",
                     method="method.AUC",
                     SL.library=sl.lib,
                     cvControl=list(V=folds, validRows=index))
# Look at coefficients and predictions
SLfitY
SLfitY$Z[,2]

# Once with screening (screen.corRank)
set.seed(123)
sl.lib <- list("SL.mean", SL.glmnet_new$names, c(SL.glmnet_new$names, "screen.corRank"))
SLfitY_scr<-SuperLearner(Y=y,X=x,family="binomial",
                         method="method.AUC",
                         SL.library=sl.lib,
                         cvControl=list(V=folds, validRows=index))
# Looking at coefficients, predictions, and which variables were selected
SLfitY_scr
SLfitY_scr$Z[,2]
SLfitY_scr$whichScreen
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# C. CODING SUPERLEARNER BY HAND
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# In my actual project, we are doing it this way to be able to include a custom algorithm 
# Here, we just want to make sure that what we're doing gives the same results as the SuperLearner function

# Fitting individual algorithms on the training set (minus the ii-th validation set)
# We had to use leave-one-out cross-validation for cv.glmnet, nfolds = 200, because the folds for cross-validation within each CV fold are chosen at random
set.seed(123)
m1 <- lapply(1:folds,function(ii) weighted.mean(rbindlist(splt[-ii])$y)) #mean - SL function uses weighted.mean
m2 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-6]), 
                                             do.call(rbind,splt[-ii])[,6], lambda=NULL, 
                                             nlambda = 100, type.measure="deviance", nfolds=200, family="binomial", alpha=1)) 

# Writing separate functions for variable selection and cv.glmnet with var selection
vsfunc <- function(ii){
  whichVar <- screen.corRank(Y=do.call(rbind,splt[-ii])[,6], X=do.call(rbind,splt[-ii])[,-6])
}
whichVar <- lapply(1:folds, function(z) vsfunc(z))


glmnet_func <- function(ii, w) {
  m3 <- cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-6][,whichVar[[ii]]]), 
                  do.call(rbind,splt[-ii])[,6], lambda=NULL, 
                  nlambda = 100, type.measure="deviance", nfolds=200, family="binomial", alpha=1)
  # return(m3)
}

m3 <- lapply(1:folds, function(z) glmnet_func(z))
# If I try to combine them into a single function, I get errors

# Use the model fits above to generate predictions in each fold
p1 <- lapply(1:folds, function(ii) rep(m1[[ii]], nrow(splt[[ii]])))
p2 <- lapply(1:folds, function(ii) predict(m2[[ii]], newx = model.matrix(~-1 + ., do.call(rbind,splt[ii])[,-6]), s="lambda.min", type="response"))

# Big question: I created whichVar on the folds that were included. Is it ok for me to now use this same vector to predict on that one that was left out?
predict_func <- function(ii) {
  p3 <- predict(m3[[ii]], newx = model.matrix(~-1 + ., do.call(rbind,splt[ii])[,-6][whichVar[[ii]]]), s="lambda.min", type="response")
  return(p3)
}

p3 <- lapply(1:folds, function(z) predict_func(z))

# Checking that predictions from what we did match SL function.
# Without variable selection:
cbind(sort(SLfitY$Z[,2]),sort(as.numeric(do.call(rbind,p2))),round(sort(SLfitY$Z[,2])-sort(as.numeric(do.call(rbind,p2))),4))
head(sort(SLfitY$Z[,2]))
head(sort(as.numeric(do.call(rbind,p2))))
# With variable selection
cbind(sort(SLfitY_scr$Z[,2]),sort(as.numeric(do.call(rbind,p3))),round(sort(SLfitY_scr$Z[,2])-sort(as.numeric(do.call(rbind,p3))),4))
head(sort(SLfitY_scr$Z[,2]))
head(sort(as.numeric(do.call(rbind,p3))))

# Updating dataframe 'splt' so that column 1 has the observed outcome (y)
# and subsequent columns contain the predictions we generated above
for(i in 1:folds){
  splt[[i]]<-cbind(splt[[i]][,6], p1[[i]], p2[[i]], p3[[i]])
}
# Looking just at the first few observations in the first fold
head(data.frame(splt[[1]]))

# Generating CV risk estimates
risk1<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2], labels=splt[[ii]][,1]))    # CV-risk for mean
risk2 <- lapply(1:folds, function(ii) 1-AUC(predictions=splt[[ii]][,3], labels=splt[[i]][,1]))  # CV-risk for glmnet all
risk3 <- lapply(1:folds, function(ii) 1-AUC(predictions=splt[[ii]][,4], labels=splt[[i]][,1]))  # CV-risk for glmnet screen corRank
# And combining them
a<-rbind(cbind("mean",mean(do.call(rbind, risk1),na.rm=T)),
         cbind("glmnet",mean(do.call(rbind, risk2),na.rm=T)),
         cbind("glmnet_select",mean(do.call(rbind,risk3), na.rm=T)))
# Also combine predicted probabilities for the metalearner
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","AC.mean","AC.glmnet_All","AC.glmnet_screen.corRank")

# Define the function we want to optimize 
bounds = c(0, Inf)
SL.r<-function(A, y, par){
  A<-as.matrix(A)
  names(par)<-c("AC.mean","AC.glmnet_All","AC.glmnet_screen.corRank")
  predictions <- crossprod(t(A),par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}

# Optimization step
init <- rep(1/3, 3)
fit <- optim(par=init, fn=SL.r, A=X[,2:4], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])

# Check that convergence was achieved
fit

# Normalize coefficients and look at them
alpha<-fit$par/sum(fit$par)
alpha

# Compare output from SL function and hand-coded function 
SLfitY_scr
alpha
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
