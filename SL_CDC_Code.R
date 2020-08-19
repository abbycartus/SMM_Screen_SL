###########################################################################################################################################################################    
###########################################################################################################################################################################    
################################################   HAND-CODING SUPERLEARNER ###############################################################################################  
################################################   ARC 7.2020               ###############################################################################################  
###########################################################################################################################################################################    
###########################################################################################################################################################################    


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# INSTALL AND LOAD PACKAGES
# READ DATA
# DEFINE OBJECTS 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","recipes","resample","caret","SuperLearner",
              "data.table","nnls","mvtnorm","ranger","xgboost","splines","Matrix","xtable","pROC","arm",
              "polspline","ROCR","cvAUC", "KernelKnn", "gam","glmnet")
for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package,repos='http://lib.stat.cmu.edu/R/CRAN') 
  }
}

#Set working directory
setwd("\\\\136.142.117.70\\Studies$\\Bodnar Abby\\Severe Maternal Morbidity\\Data")

# Read data
D <- readRDS("baked_train_momi_20200724.rds")
D_splines <- readRDS("baked_train_momi_splines_20200724.rds")

test <- readRDS("baked_test_momi_20200818.rds")
test_splines <- readRDS("baked_test_momi_splines_20200818.rds")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# VARIABLE SELECTION
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# corRank - modify to pull out the top 20 ranked varialbes (default is 2)
screen.corRank20 <- function(Y,X,family, method='pearson', rank=20){
  listp <- apply(X,2,function(x,Y,method){
    ifelse(var(x) <= 0, 1, cor.test(x,y=Y, method = method)$p.value)
  }, Y=Y, method=method)
  whichVariable <- (rank(listp)<= rank)
  return(whichVariable)
}
which.corRank20 <-  screen.corRank20(Y=D$ch_smmtrue, X=D[,-1])
which.corRank20.y <- c(ch_smmtrue=TRUE, which.corRank20)
which.corRank20_splines <-  screen.corRank20(Y=D_splines$ch_smmtrue, X=D_splines[,-1])
which.corRank20_splines.y <- c(ch_smmtrue=TRUE, which.corRank20_splines)

# corRank - modify to pull out the top 50 ranked varialbes (default is 2)
screen.corRank50 <- function(Y,X,family, method='pearson', rank=50){
  listp <- apply(X,2,function(x,Y,method){
    ifelse(var(x) <= 0, 1, cor.test(x,y=Y, method = method)$p.value)
  }, Y=Y, method=method)
  whichVariable <- (rank(listp)<= rank)
  return(whichVariable)
}
which.corRank50 <-  screen.corRank50(Y=D$ch_smmtrue, X=D[,-1])
which.corRank50.y <- c(ch_smmtrue=TRUE, which.corRank50)
which.corRank50_splines <-  screen.corRank50(Y=D_splines$ch_smmtrue, X=D_splines[,-1])
which.corRank50_splines.y <- c(ch_smmtrue=TRUE, which.corRank50_splines)


# corP
which.corP <- screen.corP(Y=D$ch_smmtrue, X=D[,-1])
which.corP.y <- c(ch_smmtrue=TRUE, which.corP)
which.corP_splines <- screen.corP(Y=D_splines$ch_smmtrue, X=D_splines[,-1])
which.corP_splines.y <- c(ch_smmtrue=TRUE, which.corP_splines)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# SPLIT DATA INTO 10 FOLDS FOR 10-FOLD CV 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Variable selection
#D_splt <- D[,which.corRank50.y]
#D_splt_splines <- D[,which.corRank50_splines.y]

folds=10
index<-split(1:nrow(D),1:folds)

splt<-lapply(1:folds,function(ind) D[index[[ind]],])
splt_splines <- lapply(1:folds, function(ind) D_splines[index[[ind]],])

index_test <- split(1:nrow(test), 1:folds)
sptest <- lapply(1:folds, function(ind) test[index_test[[ind]],])
sptest_splines <- lapply(1:folds, function(ind) test_splines[index_test[[ind]],])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# CODE INDIVIDUAL MODELS
# PREDICT FROM INDIVIDUAL MODELS 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
## We are fitting individual algorithms on the training set, excluding the iith validation set

set.seed(123)

# bayesglm with defaults
m1<-lapply(1:folds,function(ii) bayesglm(formula=ch_smmtrue~.,data=do.call(rbind,splt[-ii]),family="binomial")) 
#random forest (ranger) with a range of tuning parameters
m2 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = T))
m3 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 3, min.node.size = 10, replace = T))
m4 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 4, min.node.size = 10, replace = T))
m5 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = F))
m6 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 3, min.node.size = 10, replace = F))
m7 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 4, min.node.size = 10, replace = F))
m8 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 2, min.node.size = 10, replace = T))
m9 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 3, min.node.size = 10, replace = T))
m10 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 4, min.node.size = 10, replace = T))
m11 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 2, min.node.size = 10, replace = F))
m12 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 3, min.node.size = 10, replace = F))
m13 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 4, min.node.size = 10, replace = F))
#mean
m14 <- lapply(1:folds,function(ii) mean(rbindlist(splt[-ii])$ch_smmtrue))
#glm
m15 <- lapply(1:folds, function(ii) glm(ch_smmtrue~., data=do.call(rbind,splt[-ii]), family="binomial"))
#gams - make sure to use splt_splines 
m16 <- lapply(1:folds,function(ii) gam(ch_smmtrue~., family="binomial",data=do.call(rbind,splt_splines[-ii])))
#glmnet - also vary lambdas? 
#cv.glmnet will find the optimal lambda for you... can you incorporate cross-validation of lambda into CV we're already doing 
#also - we standardized variables before in preprocessing, so need to set standardize = FALSE 
#m17 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), do.call(rbind,splt[-ii])[,9]), alpha = 0, family="binomial",   standardize = FALSE)
#m18 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), do.call(rbind,splt[-ii])[,9]), alpha = 0.2, family="binomial", standardize = FALSE)
#m19 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), do.call(rbind,splt[-ii])[,9]), alpha = 0.4, family="binomial", standardize = FALSE)
#m20 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), do.call(rbind,splt[-ii])[,9]), alpha = 0.6, family="binomial", standardize = FALSE)
#m21 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), do.call(rbind,splt[-ii])[,9]), alpha = 0.8, family="binomial", standardize = FALSE)
#m22 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), do.call(rbind,splt[-ii])[,9]), alpha = 1.0, family="binomial", standardize = FALSE)

# changed cv.glmnet to give us closer to what we would get with superlearner, based on var selection example.
# tried to do nfolds = nrow(splt[ii]) but that didn't work; set all to 68 (all follds have 68 or 69 rows)
m17 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), as.matrix(do.call(rbind,splt[-ii])[,9]), 
                                             lambda=NULL,nlambda = 100, type.measure = "deviance",  nfolds=68, family="binomial", alpha=0))
m18 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), as.matrix(do.call(rbind,splt[-ii])[,9]), 
                                             lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=68, family="binomial", alpha=0.2))
m19 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), as.matrix(do.call(rbind,splt[-ii])[,9]), 
                                             lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=68, family="binomial", alpha=0.4))
m20 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), as.matrix(do.call(rbind,splt[-ii])[,9]), 
                                             lambda=NULL,nlambda = 100, type.measure = "deviance",  nfolds=68, family="binomial", alpha=0.6))
m21 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), as.matrix(do.call(rbind,splt[-ii])[,9]), 
                                             lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=68, family="binomial", alpha=0.8))
m22 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-9]), as.matrix(do.call(rbind,splt[-ii])[,9]), 
                                             lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=68, family="binomial", alpha=1))


# k-neaest neighbors: build prediction into model statement with TEST_data; yields a list of predicted probabilities directly 
p23 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-9],y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,9]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-9], regression=T, k=2))
p24 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-9], y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,9]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-9],regression=T, k=3))
p25 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-9], y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,9]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-9],regression=T, k=4))
p26 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-9], y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,9]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-9], regression=T, k=5))
#cdc
#CDC algorithm: just classify as 0 or 1 on the basis of the SMM screening criteria. Need to double check that the numbers you get here match the screening var.
p27 <- lapply(1:folds, function(ii) do.call(rbind,splt[ii]) %>% 
                mutate(cdc_screen = ifelse(mmortanestcomp_X1==1 | mmortbloodtrans_X1==1 | mmortcardia_X1==1 |
                                             mmortcereb_X1==1 | mmortcoag_X1==1 | mmorteclamp_X1==1 | mmorthyster_X1==1 | mmortpuledema_X1==1 | mmortrenal_X1==1 |
                                             mmortrespdis_X1==1 | mmortsepsis_X1==1 | mmortshock_X1==1 | mmortsicklecell_X1==1 | mmortthomembol_X1==1 | mmortvent_X1==1 | 
                                             icuadmit_Yes==1 | los_3sd_Yes==1, 1, 0)) %>% 
                dplyr::select(c(cdc_screen)))


## Now, obtain the predicted probability of the outcome for observation in the ii-th validation set
########## BAYESGLM ##########  
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=do.call(rbind,splt[ii]),type="response"))

########## RANGER ##########   
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],data=do.call(rbind,splt[ii])))
p3<-lapply(1:folds,function(ii) predict(m3[[ii]],data=do.call(rbind,splt[ii])))
p4<-lapply(1:folds,function(ii) predict(m4[[ii]],data=do.call(rbind,splt[ii])))
p5<-lapply(1:folds,function(ii) predict(m5[[ii]],data=do.call(rbind,splt[ii])))
p6<-lapply(1:folds,function(ii) predict(m6[[ii]],data=do.call(rbind,splt[ii])))
p7<-lapply(1:folds,function(ii) predict(m7[[ii]],data=do.call(rbind,splt[ii])))
p8<-lapply(1:folds,function(ii) predict(m8[[ii]],data=do.call(rbind,splt[ii])))
p9<-lapply(1:folds,function(ii) predict(m9[[ii]],data=do.call(rbind,splt[ii])))
p10<-lapply(1:folds,function(ii) predict(m10[[ii]],data=do.call(rbind,splt[ii])))
p11<-lapply(1:folds,function(ii) predict(m11[[ii]],data=do.call(rbind,splt[ii])))
p12<-lapply(1:folds,function(ii) predict(m12[[ii]],data=do.call(rbind,splt[ii])))
p13<-lapply(1:folds,function(ii) predict(m13[[ii]],data=do.call(rbind,splt[ii])))

########## MEAN ##########  
p14 <- lapply(1:folds, function(ii) rep(m14[[ii]], nrow(splt[[ii]])))

########## GLM ##########  
# I removed the double brackets from splt[[ii]] and now the command works... 
p15 <- lapply(1:folds, function(ii) predict(m15[[ii]], newdata = do.call(rbind,splt[ii]), type="respo"))

########## GAMS ##########   
p16 <- lapply(1:folds, function(ii) predict(m16[[ii]], newdata=do.call(rbind,splt_splines[ii]), type="response"))

########## GLMNET ########## 
p17 <- lapply(1:folds, function(ii) predict(m17[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-1]), s="lambda.min", type="response"))
p18 <- lapply(1:folds, function(ii) predict(m18[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-1]), s="lambda.min", type="response"))
p19 <- lapply(1:folds, function(ii) predict(m19[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-1]), s="lambda.min", type="response"))
p20 <- lapply(1:folds, function(ii) predict(m20[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-1]), s="lambda.min", type="response"))
p21 <- lapply(1:folds, function(ii) predict(m21[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-1]), s="lambda.min", type="response"))
p22 <- lapply(1:folds, function(ii) predict(m22[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-1]), s="lambda.min", type="response"))
#you're seeing predictions for different values of lambda. we're supposed to pick an optimal value of lambda. typically picked with cross-validation.

########## KNN: PREDICTED PROBABILITIES ARE ALREADY IN P50-P53 ########## 
########## CDC: ALREADY DONE ########## 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# COMBINE PREDICTIONS FROM ABOVE MODELS INTO ONE DATAFRAME 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# update dataframe 'splt' so that column1 is the observed outcome (y) and subsequent columns contain predictions above

for(i in 1:folds){
  splt[[i]]<-cbind(splt[[i]][,9],
                   p1[[i]], #bayesglm
                   p2[[i]]$predictions, p3[[i]]$predictions, p4[[i]]$predictions, p5[[i]]$predictions, p6[[i]]$predictions, p7[[i]]$predictions, p8[[i]]$predictions,
                    p9[[i]]$predictions, p10[[i]]$predictions, p11[[i]]$predictions, p12[[i]]$predictions, p13[[i]]$predictions, #ranger
                   p14[[i]], #mean
                   p15[[i]], #glm
                   p16[[i]], #gams
                   p17[[i]], p18[[i]], p19[[i]], p20[[i]], p21[[i]], p22[[i]], #glmnet
                   p23[[i]], p24[[i]], p25[[i]], p26[[i]], #knn
                   p27[[i]]) #CDC
}
# view the first 6 observations in the first fold 
head(data.frame(splt[[1]]))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE CV RISK FOR EACH METHOD
# AVERAGE CV RISK OVER ALL 10 FOLDS TO GET 1 PERFORMANCE MEASURE PER ALGORITHM
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# First, calculate CV risk for each method for the ii-th validation set
# our loss function is the rank loss; so our risk is (1-AUC)
#	use the AUC() function with input as the predicted outcomes and 'labels' as the true outcomes
risk1<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2], labels=splt[[ii]][,1]))    # CV-risk for bayesglm
risk2<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,3], labels=splt[[ii]][,1]))		# CV-risk for ranger 1
risk3<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,4], labels=splt[[ii]][,1]))		# CV-risk for ranger 2
risk4<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,5], labels=splt[[ii]][,1]))		# CV-risk for ranger 3
risk5<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,6], labels=splt[[ii]][,1]))		# CV-risk for ranger 4
risk6<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,7], labels=splt[[ii]][,1]))		# CV-risk for ranger 5
risk7<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,8], labels=splt[[ii]][,1]))		# CV-risk for ranger 6
risk8<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,9], labels=splt[[ii]][,1]))		# CV-risk for ranger 7
risk9<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,10], labels=splt[[ii]][,1]))		# CV-risk for ranger 8
risk10<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,11], labels=splt[[ii]][,1]))		# CV-risk for ranger 9
risk11<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,12], labels=splt[[ii]][,1]))		# CV-risk for ranger 10
risk12<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,13], labels=splt[[ii]][,1]))		# CV-risk for ranger 11
risk13<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,14], labels=splt[[ii]][,1]))		# CV-risk for ranger 12
risk14<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,15], labels=splt[[ii]][,1]))		# CV-risk for mean
risk15<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,16], labels=splt[[ii]][,1]))		# CV-risk for glm
risk16<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,17], labels=splt[[ii]][,1]))		# CV-risk for gams
risk17<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,17], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk18<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,18], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk19<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,19], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk20<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,20], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk21<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,21], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk22<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,22], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk23<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,23], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk24<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,24], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk25<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,25], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk26<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,26], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk27<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,27], labels=splt[[ii]][,1]))		# CV-risk for CDC

# Next, average the estimated 5 risks across the folds to obtain 1 measure of performance for each algorithm
a<-rbind(cbind("bayesglm",mean(do.call(rbind, risk1),na.rm=T)),
         cbind("ranger1",mean(do.call(rbind, risk2),na.rm=T)),
         cbind("ranger2",mean(do.call(rbind,risk3), na.rm=T)),
         cbind("ranger3",mean(do.call(rbind,risk4), na.rm=T)),
         cbind("ranger4",mean(do.call(rbind,risk5), na.rm=T)),
         cbind("ranger5",mean(do.call(rbind,risk6), na.rm=T)),
         cbind("ranger6",mean(do.call(rbind,risk7), na.rm=T)),
         cbind("ranger7",mean(do.call(rbind,risk8), na.rm=T)),
         cbind("ranger8",mean(do.call(rbind,risk9), na.rm=T)),
         cbind("ranger9",mean(do.call(rbind,risk10), na.rm=T)),
         cbind("ranger10",mean(do.call(rbind,risk11), na.rm=T)),
         cbind("ranger11",mean(do.call(rbind,risk12), na.rm=T)),
         cbind("ranger12",mean(do.call(rbind,risk13), na.rm=T)),
         cbind("mean",mean(do.call(rbind,risk14), na.rm=T)),
         cbind("glm",mean(do.call(rbind,risk15), na.rm=T)),
         cbind("gam",mean(do.call(rbind,risk16), na.rm=T)),
         cbind("glmnet1",mean(do.call(rbind,risk17), na.rm=T)),
         cbind("glmnet2",mean(do.call(rbind,risk18), na.rm=T)),
         cbind("glmnet3",mean(do.call(rbind,risk19), na.rm=T)),
         cbind("glmnet4",mean(do.call(rbind,risk20), na.rm=T)),
         cbind("glmnet5",mean(do.call(rbind,risk21), na.rm=T)),
         cbind("glmnet6",mean(do.call(rbind,risk22), na.rm=T)),
         cbind("knn1",mean(do.call(rbind,risk23), na.rm=T)),
         cbind("knn2",mean(do.call(rbind,risk24), na.rm=T)),
         cbind("knn3",mean(do.call(rbind,risk25), na.rm=T)),
         cbind("knn4",mean(do.call(rbind,risk26), na.rm=T)),
         cbind("cdc",mean(do.call(rbind,risk27), na.rm=T)))

# this is one part of "SuperLearner" output. How do we get 95% CI's around these so we can plot them nicely?
saveRDS(a, "risks_momi_noscreening_20200731.rds")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# ESTIMATE SUPERLEARNER WEIGHTS BY MINIMIIZNG 1-auc
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# here: stimate SL weights using the optim() function to minimize (1-AUC)
# turn splt into a data frame (X) and define names 
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","bayesglm",
                                                                "ranger1","ranger2","ranger3","ranger4","ranger5","ranger6","ranger7",
                                                                "ranger8","ranger9","ranger10","ranger11","ranger12",
                                                                "mean",
                                                                "glm",
                                                                "gams",
                                                                "glmnet1","glmnet2","glmnet3","glmnet4","glmnet5","glmnet6",
                                                                "knn1","knn2","knn3","knn4",
                                                                "cdc")
head(X)

# Define the function we want to optimize (SL.r)
SL.r<-function(A, y, par){
  A<-as.matrix(A)
  names(par)<-c("bayesglm",
                "ranger1","ranger2","ranger3","ranger4","ranger5","ranger6","ranger7","ranger8",
                "ranger9","ranger10","ranger11","ranger12",
                "mean",
                "glm",
                "gams",
                "glmnet1","glmnet2","glmnet3",
                "glmnet4","glmnet5","glmnet6",
                "knn1","knn2","knn3","knn4",
                "cdc")
  predictions <- crossprod(t(A),par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}


# Define bounds and starting values
# init should be 1/par, par where par is number of predictors (excluding y)
bounds = c(0, Inf)
init <- rep(1/27, 27)

# Optimize SL.r
fit <- optim(par=init, fn=SL.r, A=X[,2:28], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])
fit

# Normalize the coefficients and look at them
alpha<-fit$par/sum(fit$par)
options(scipen=999) #changes from scientific to numeric notation 
alpha
sum(alpha)

# If you want to save the coefficients you can:
alpha_data <- data.frame(alpha)
rownames(alpha_data) <- c("bayesglm","ranger1","ranger2","ranger3","ranger4","ranger5","ranger6","ranger7","ranger8",
                "ranger9","ranger10","ranger11","ranger12","mean","glm","gams","glmnet1","glmnet2","glmnet3",
               "glmnet4","glmnet5","glmnet6","knn1","knn2","knn3","knn4","cdc")

write.csv(alpha_data, file="C:\\Users\\abc72\\Box Sync\\Dissertation backup\\Dissertation\\MOMI\\Exploratory data analysis\\alpha_varselection_noscreening_all_2020.07.31.csv")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# RE-FIT ALL ALGORITHMS TO NEW DATA 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

set.seed(123)

# bayesglm with defaults
m1<-lapply(1:folds,function(ii) bayesglm(formula=ch_smmtrue~.,data=do.call(rbind,sptest[-ii]),family="binomial")) 
#random forest (ranger) with a range of tuning parameters
m2 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = T))
m3 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 500, mtry = 3, min.node.size = 10, replace = T))
m4 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 500, mtry = 4, min.node.size = 10, replace = T))
m5 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = F))
m6 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 500, mtry = 3, min.node.size = 10, replace = F))
m7 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 500, mtry = 4, min.node.size = 10, replace = F))
m8 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 2500, mtry = 2, min.node.size = 10, replace = T))
m9 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 2500, mtry = 3, min.node.size = 10, replace = T))
m10 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 2500, mtry = 4, min.node.size = 10, replace = T))
m11 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 2500, mtry = 2, min.node.size = 10, replace = F))
m12 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 2500, mtry = 3, min.node.size = 10, replace = F))
m13 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), num.trees = 2500, mtry = 4, min.node.size = 10, replace = F))
#mean
m14 <- lapply(1:folds,function(ii) mean(rbindlist(sptest[-ii])$ch_smmtrue))
#glm
m15 <- lapply(1:folds, function(ii) glm(ch_smmtrue~., data=do.call(rbind,sptest[-ii]), family="binomial"))
#gams - make sure to use sptest_splines 
m16 <- lapply(1:folds,function(ii) gam(ch_smmtrue~., family="binomial",data=do.call(rbind,sptest_splines[-ii])))
#glmnet - also vary lambdas? 
#cv.glmnet will find the optimal lambda for you... can you incorporate cross-validation of lambda into CV we're already doing 
#also - we standardized variables before in preprocessing, so need to set standardize = FALSE 
#m17 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), do.call(rbind,sptest[-ii])[,9]), alpha = 0, family="binomial",   standardize = FALSE)
#m18 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), do.call(rbind,sptest[-ii])[,9]), alpha = 0.2, family="binomial", standardize = FALSE)
#m19 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), do.call(rbind,sptest[-ii])[,9]), alpha = 0.4, family="binomial", standardize = FALSE)
#m20 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), do.call(rbind,sptest[-ii])[,9]), alpha = 0.6, family="binomial", standardize = FALSE)
#m21 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), do.call(rbind,sptest[-ii])[,9]), alpha = 0.8, family="binomial", standardize = FALSE)
#m22 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), do.call(rbind,sptest[-ii])[,9]), alpha = 1.0, family="binomial", standardize = FALSE)

# changed cv.glmnet to give us closer to what we would get with superlearner, based on var selection example.
# tried to do nfolds = nrow(sptest[ii]) but that didn't work; set all to 45 (all follds have 45 or 46 rows)
m17 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), as.matrix(do.call(rbind,sptest[-ii])[,9]), 
                                              lambda=NULL,nlambda = 100, type.measure = "deviance",  nfolds=45, family="binomial", alpha=0))
m18 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), as.matrix(do.call(rbind,sptest[-ii])[,9]), 
                                              lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=45, family="binomial", alpha=0.2))
m19 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), as.matrix(do.call(rbind,sptest[-ii])[,9]), 
                                              lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=45, family="binomial", alpha=0.4))
m20 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), as.matrix(do.call(rbind,sptest[-ii])[,9]), 
                                              lambda=NULL,nlambda = 100, type.measure = "deviance",  nfolds=45, family="binomial", alpha=0.6))
m21 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), as.matrix(do.call(rbind,sptest[-ii])[,9]), 
                                              lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=45, family="binomial", alpha=0.8))
m22 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,sptest[-ii])[,-9]), as.matrix(do.call(rbind,sptest[-ii])[,9]), 
                                              lambda=NULL, nlambda = 100, type.measure = "deviance", nfolds=45, family="binomial", alpha=1))

# k-neaest neighbors: build prediction into model statement with TEST_data; yields a list of predicted probabilities directly 
p23 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,sptest[-ii])[,-9],y=as.numeric(as.character(unlist(do.call(rbind,sptest[-ii])[,9]))),
                                              TEST_data = do.call(rbind,sptest[ii])[,-9], regression=T, k=2))
p24 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,sptest[-ii])[,-9], y=as.numeric(as.character(unlist(do.call(rbind,sptest[-ii])[,9]))),
                                              TEST_data = do.call(rbind,sptest[ii])[,-9],regression=T, k=3))
p25 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,sptest[-ii])[,-9], y=as.numeric(as.character(unlist(do.call(rbind,sptest[-ii])[,9]))),
                                              TEST_data = do.call(rbind,sptest[ii])[,-9],regression=T, k=4))
p26 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,sptest[-ii])[,-9], y=as.numeric(as.character(unlist(do.call(rbind,sptest[-ii])[,9]))),
                                              TEST_data = do.call(rbind,sptest[ii])[,-9], regression=T, k=5))
#cdc
#CDC algorithm: just classify as 0 or 1 on the basis of the SMM screening criteria. Need to double check that the numbers you get here match the screening var.
p27 <- lapply(1:folds, function(ii) do.call(rbind,sptest[ii]) %>% 
                mutate(cdc_screen = ifelse(mmortanestcomp_X1==1 | mmortbloodtrans_X1==1 | mmortcardia_X1==1 |
                                             mmortcereb_X1==1 | mmortcoag_X1==1 | mmorteclamp_X1==1 | mmorthyster_X1==1 | mmortpuledema_X1==1 | mmortrenal_X1==1 |
                                             mmortrespdis_X1==1 | mmortsepsis_X1==1 | mmortshock_X1==1 | mmortsicklecell_X1==1 | mmortthomembol_X1==1 | mmortvent_X1==1 | 
                                             icuadmit_Yes==1 | los_3sd_Yes==1, 1, 0)) %>% 
                dplyr::select(c(cdc_screen)))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREDICT PROBABILITIES FROM EACH FIT USING ALL TEST DATA 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
## Now, obtain the predicted probability of the outcome for observation in the ii-th validation set
########## BAYESGLM ##########  
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=do.call(rbind,sptest[ii]),type="response"))

########## RANGER ##########   
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],data=do.call(rbind,sptest[ii])))
p3<-lapply(1:folds,function(ii) predict(m3[[ii]],data=do.call(rbind,sptest[ii])))
p4<-lapply(1:folds,function(ii) predict(m4[[ii]],data=do.call(rbind,sptest[ii])))
p5<-lapply(1:folds,function(ii) predict(m5[[ii]],data=do.call(rbind,sptest[ii])))
p6<-lapply(1:folds,function(ii) predict(m6[[ii]],data=do.call(rbind,sptest[ii])))
p7<-lapply(1:folds,function(ii) predict(m7[[ii]],data=do.call(rbind,sptest[ii])))
p8<-lapply(1:folds,function(ii) predict(m8[[ii]],data=do.call(rbind,sptest[ii])))
p9<-lapply(1:folds,function(ii) predict(m9[[ii]],data=do.call(rbind,sptest[ii])))
p10<-lapply(1:folds,function(ii) predict(m10[[ii]],data=do.call(rbind,sptest[ii])))
p11<-lapply(1:folds,function(ii) predict(m11[[ii]],data=do.call(rbind,sptest[ii])))
p12<-lapply(1:folds,function(ii) predict(m12[[ii]],data=do.call(rbind,sptest[ii])))
p13<-lapply(1:folds,function(ii) predict(m13[[ii]],data=do.call(rbind,sptest[ii])))

########## MEAN ##########  
p14 <- lapply(1:folds, function(ii) rep(m14[[ii]], nrow(sptest[[ii]])))

########## GLM ##########  
# I removed the double brackets from sptest[[ii]] and now the command works... 
p15 <- lapply(1:folds, function(ii) predict(m15[[ii]], newdata = do.call(rbind,sptest[ii]), type="respo"))

########## GAMS ##########   
p16 <- lapply(1:folds, function(ii) predict(m16[[ii]], newdata=do.call(rbind,sptest_splines[ii]), type="response"))

########## GLMNET ########## 
p17 <- lapply(1:folds, function(ii) predict(m17[[ii]], newx = as.matrix(do.call(rbind,sptest[ii])[,-1]), s="lambda.min", type="response"))
p18 <- lapply(1:folds, function(ii) predict(m18[[ii]], newx = as.matrix(do.call(rbind,sptest[ii])[,-1]), s="lambda.min", type="response"))
p19 <- lapply(1:folds, function(ii) predict(m19[[ii]], newx = as.matrix(do.call(rbind,sptest[ii])[,-1]), s="lambda.min", type="response"))
p20 <- lapply(1:folds, function(ii) predict(m20[[ii]], newx = as.matrix(do.call(rbind,sptest[ii])[,-1]), s="lambda.min", type="response"))
p21 <- lapply(1:folds, function(ii) predict(m21[[ii]], newx = as.matrix(do.call(rbind,sptest[ii])[,-1]), s="lambda.min", type="response"))
p22 <- lapply(1:folds, function(ii) predict(m22[[ii]], newx = as.matrix(do.call(rbind,sptest[ii])[,-1]), s="lambda.min", type="response"))
#you're seeing predictions for different values of lambda. we're supposed to pick an optimal value of lambda. typically picked with cross-validation.


for(i in 1:folds){
  sptest[[i]]<-cbind(
                   p1[[i]], #bayesglm
                   p2[[i]]$predictions, p3[[i]]$predictions, p4[[i]]$predictions, p5[[i]]$predictions, p6[[i]]$predictions, p7[[i]]$predictions, p8[[i]]$predictions,
                    p9[[i]]$predictions, p10[[i]]$predictions, p11[[i]]$predictions, p12[[i]]$predictions, p13[[i]]$predictions, #ranger
                   p14[[i]], #mean
                   p15[[i]], #glm
                   p16[[i]], #gams
                   p17[[i]], p18[[i]], p19[[i]], p20[[i]], p21[[i]], p22[[i]], #glmnet
                   p23[[i]], p24[[i]], p25[[i]], p26[[i]], #knn
                   p27[[i]]) #CDC
}
# view the first 6 observations in the first fold 
head(data.frame(sptest[[1]]))

predictions <- do.call(rbind, sptest)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# TAKE A WEIGHTED COMBINATION OF PREDICTIONS USING NNLS COEFFICIENTS AS WEIGHTS 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
predmat <- as.matrix(predictions)
alpha <- as.matrix(alpha)
y_pred <- predmat%*%alpha

p<-data.frame(y=test$ch_smmtrue,y_pred=y_pred)

# quick visualization
momi_roc <- roc(p[,1], p[,2], direction = "auto")
plot_el <- data.frame(sens = momi_roc$sensitivities, spec = momi_roc$specificities)

cdc_roc <- roc(test$ch_smmtrue, predictions$cdc_screen, direction = "auto")
plot_cdc <- data.frame(sens = cdc_roc$sensitivities, spec = cdc_roc$specificities)

ggplot() + 
  geom_step(data = plot_el, aes(1-spec, sens), linetype = 2, size = 0.5)+
  geom_step(data = plot_cdc, aes(1-spec, sens), linetype = 2, size = 0.5) +
    theme_light() + theme(legend.position=c(.8,.2)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) + 
  labs(x = "1 - Specificity",y = "Sensitivity") +
  geom_abline(intercept=0,slope=1,col="gray") +
  scale_colour_manual(name="",values=cols)

sum(test$ch_smmtrue)
sum(predictions$cdc_screen)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# VISUALIZE RESULTS  
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# fits from candidate algorithms
a<-roc(D$ch_smmtrue, p1, direction="auto")
Cbayes<-data.frame(sens=a$sensitivities,spec=a$specificities)

a<-roc(D$ch_smmtrue, p2, direction="auto") 
Cpoly<-data.frame(sens=a$sensitivities,spec=a$specificities)

cols <- c("Bayes GLM"="green", "PolyMARS"="black")
ggplot() +
  geom_step(data=Cbayes, aes(1-spec,sens,color="Bayes GLM"),linetype=2,size=.5) +
  geom_step(data=Cpoly, aes(1-spec,sens,color="PolyMARS"),linetype=2,size=.5) +
  #scale_colour_manual(name="",values=cols) +
  theme_light() + theme(legend.position=c(.8,.2)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) + 
  labs(x = "1 - Specificity",y = "Sensitivity") +
  geom_abline(intercept=0,slope=1,col="gray") +
  scale_colour_manual(name="",values=cols)



