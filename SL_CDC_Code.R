# Adapting AN & LB's code to MOMI data 

# load the relevant packages
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

# Specify the number of folds for V-fold cross-validation
folds=10

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hand-coding Super Learner
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
## 1: split data into 10 groups for 10-fold cross-validation 
splt<-split(D,1:folds)

splt_splines <- split(D_splines, 1:folds)

## 2: Fitting individual algorithms on the training set (but not the ii-th validation set)
set.seed(123)
# bayesglm with defaults
m1<-lapply(1:folds,function(ii) bayesglm(formula=ch_smmtrue~.,data=do.call(rbind,splt[-ii]),family="binomial")) #bayesglm
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
m16 <- lapply(1:folds,function(ii) gam(ch_smmtrue~., family="binomial",data=rbindlist(splt_splines[-ii])))

#xgboost
m17 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=200, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m18 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=200, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m19 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=200, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m20 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=500, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m21 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=500, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m22 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=500, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m23 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=1000, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m24 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=1000, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m25 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=1000, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m26 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=200, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m27 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=200, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m28 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=200, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m29 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=500, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m30 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=500, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m31 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=500, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m32 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=1000, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m33 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=1000, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m34 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=1000, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m35 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=200, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m36 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=200, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m37 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=200, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m38 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=500, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m39 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=500, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m40 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=500, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m41 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 4, ntrees=1000, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m42 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 5, ntrees=1000, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m43 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-1]),label = as.matrix(do.call(rbind,splt[-ii])[,1]), max_depth = 6, ntrees=1000, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))

#glmnet - also vary lambdas? 
#cv.glmnet will find the optimal lambda for you... can you incorporate cross-validation of lambda into CV we're already doing 
m44 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-1]), as.matrix(do.call(rbind,splt[-ii])[,1]), alpha = 0, family="binomial"))
m45 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-1]), as.matrix(do.call(rbind,splt[-ii])[,1]), alpha = 0.2, family="binomial"))
m46 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-1]), as.matrix(do.call(rbind,splt[-ii])[,1]), alpha = 0.4, family="binomial"))
m47 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-1]), as.matrix(do.call(rbind,splt[-ii])[,1]), alpha = 0.6, family="binomial"))
m48 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-1]), as.matrix(do.call(rbind,splt[-ii])[,1]), alpha = 0.8, family="binomial"))
m49 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-1]), as.matrix(do.call(rbind,splt[-ii])[,1]), alpha = 1.0, family="binomial"))

# k-neaest neighbors: build prediction into model statement with TEST_data; yields a list of predicted probabilities directly 
p50 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-1],y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,1]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-1], regression=T, k=2))

p51 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-1], y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,1]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-1],regression=T, k=3))

p52 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-1], y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,1]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-1],regression=T, k=4))

p53 <- lapply(1:folds, function(ii) KernelKnn(data=do.call(rbind,splt[-ii])[,-1], y=as.numeric(as.character(unlist(do.call(rbind,splt[-ii])[,1]))),
                                              TEST_data = do.call(rbind,splt[ii])[,-1], regression=T, k=5))
 

#CDC algorithm: just classify as 0 or 1 on the basis of the SMM screening criteria. Need to double check that the numbers you get here match the screening var.
p54 <- lapply(1:folds, function(ii) do.call(rbind,splt[ii]) %>% 
               mutate(cdc_screen = ifelse(mmortafembo_X1==1 | mmortanestcomp_X1==1 | mmortaneurysm_X1==1 | mmortbloodtrans_X1==1 | mmortcardia_X1==1 | mmortcardiacrhyth_X1==1 |
                         mmortcereb_X1==1 | mmortcoag_X1==1 | mmorteclamp_X1==1 | mmortheartfail_X1==1 | mmorthyster_X1==1 | mmortpuledema_X1==1 | mmortrenal_X1==1 |
                         mmortrespdis_X1==1 | mmortsepsis_X1==1 | mmortshock_X1==1 | mmortsicklecell_X1==1 | mmortthomembol_X1==1 | mmortvent_X1==1 | 
                         M_icuadmit_Yes==1 | M_los_3sd_Yes==1, 1, 0)) %>% 
                dplyr::select(c(cdc_screen)))


## 2c: obtain the predicted probability of the outcome for observation in the ii-th validation set
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
########## BAYESGLM ##########  
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=rbindlist(splt[ii]),type="response"))

########## RANGER ##########   
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],data=rbindlist(splt[ii])))
p3<-lapply(1:folds,function(ii) predict(m3[[ii]],data=rbindlist(splt[ii])))
p4<-lapply(1:folds,function(ii) predict(m4[[ii]],data=rbindlist(splt[ii])))
p5<-lapply(1:folds,function(ii) predict(m5[[ii]],data=rbindlist(splt[ii])))
p6<-lapply(1:folds,function(ii) predict(m6[[ii]],data=rbindlist(splt[ii])))
p7<-lapply(1:folds,function(ii) predict(m7[[ii]],data=rbindlist(splt[ii])))
p8<-lapply(1:folds,function(ii) predict(m8[[ii]],data=rbindlist(splt[ii])))
p9<-lapply(1:folds,function(ii) predict(m9[[ii]],data=rbindlist(splt[ii])))
p10<-lapply(1:folds,function(ii) predict(m10[[ii]],data=rbindlist(splt[ii])))
p11<-lapply(1:folds,function(ii) predict(m11[[ii]],data=rbindlist(splt[ii])))
p12<-lapply(1:folds,function(ii) predict(m12[[ii]],data=rbindlist(splt[ii])))
p13<-lapply(1:folds,function(ii) predict(m13[[ii]],data=rbindlist(splt[ii])))

########## MEAN ##########  
p14 <- lapply(1:folds, function(ii) rep(m14[[ii]], nrow(splt[[ii]])))

########## GLM ##########  
# I removed the double brackets from splt[[ii]] and now the command works... 
p15 <- lapply(1:folds, function(ii) predict(m15[[ii]], newdata = rbindlist(splt[ii]), type="respo"))

########## GAMS ##########   
p16 <- lapply(1:folds, function(ii) predict(m16[[ii]], newdata=rbindlist(splt_splines[ii]), type="response"))

########## XGBOOST ########## 
p17 <- lapply(1:folds, function(ii) predict(m17[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p18 <- lapply(1:folds, function(ii) predict(m18[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p19 <- lapply(1:folds, function(ii) predict(m19[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p20 <- lapply(1:folds, function(ii) predict(m20[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p21 <- lapply(1:folds, function(ii) predict(m21[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p22 <- lapply(1:folds, function(ii) predict(m22[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p23 <- lapply(1:folds, function(ii) predict(m23[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p24 <- lapply(1:folds, function(ii) predict(m24[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p25 <- lapply(1:folds, function(ii) predict(m25[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p26 <- lapply(1:folds, function(ii) predict(m26[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p27 <- lapply(1:folds, function(ii) predict(m27[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p28 <- lapply(1:folds, function(ii) predict(m28[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p29 <- lapply(1:folds, function(ii) predict(m29[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p30 <- lapply(1:folds, function(ii) predict(m30[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p31 <- lapply(1:folds, function(ii) predict(m31[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p32 <- lapply(1:folds, function(ii) predict(m32[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p33 <- lapply(1:folds, function(ii) predict(m33[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p34 <- lapply(1:folds, function(ii) predict(m34[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p35 <- lapply(1:folds, function(ii) predict(m35[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p36 <- lapply(1:folds, function(ii) predict(m36[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p37 <- lapply(1:folds, function(ii) predict(m37[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p38 <- lapply(1:folds, function(ii) predict(m38[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p39 <- lapply(1:folds, function(ii) predict(m39[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p40 <- lapply(1:folds, function(ii) predict(m40[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p41 <- lapply(1:folds, function(ii) predict(m41[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p42 <- lapply(1:folds, function(ii) predict(m42[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 
p43 <- lapply(1:folds, function(ii) predict(m43[[ii]], as.matrix(rbindlist(splt[ii])[,-1]))) 

########## GLMNET ########## 
p44 <- lapply(1:folds, function(ii) predict(m44[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-1]), s="lambda.min", type="response"))
p45 <- lapply(1:folds, function(ii) predict(m45[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-1]), s="lambda.min", type="response"))
p46 <- lapply(1:folds, function(ii) predict(m46[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-1]), s="lambda.min", type="response"))
p47 <- lapply(1:folds, function(ii) predict(m47[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-1]), s="lambda.min", type="response"))
p48 <- lapply(1:folds, function(ii) predict(m48[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-1]), s="lambda.min", type="response"))
p49 <- lapply(1:folds, function(ii) predict(m49[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-1]), s="lambda.min", type="response"))
#you're seeing predictions for different values of lambda. we're supposed to pick an optimal value of lambda. typically picked with cross-validation.

########## KNN: PREDICTED PROBABILITIES ARE ALREADY IN P50-P53 ########## 
########## CDC: ALREADY DONE ########## 



# update dataframe 'splt' so that column1 is the observed outcome (y) and subsequent columns contain predictions above

for(i in 1:folds){
  splt[[i]]<-cbind(splt[[i]][,1],
                   p1[[i]], #bayesglm
                   p2[[i]]$predictions, p3[[i]]$predictions, p4[[i]]$predictions, p5[[i]]$predictions, p6[[i]]$predictions, p7[[i]]$predictions, p8[[i]]$predictions,
                    p9[[i]]$predictions, p10[[i]]$predictions, p11[[i]]$predictions, p12[[i]]$predictions, p13[[i]]$predictions, #ranger
                   p14[[i]], #mean
                   p15[[i]], #glm
                   p16[[i]], #gams
                   p17[[i]], p18[[i]], p19[[i]], p20[[i]], p21[[i]], p22[[i]], p23[[i]], p24[[i]], p25[[i]], p26[[i]], p27[[i]], p28[[i]], p29[[i]], p30[[i]], p31[[i]],
                    p32[[i]], p33[[i]], p34[[i]], p35[[i]], p36[[i]], p37[[i]], p38[[i]], p39[[i]], p40[[i]], p41[[i]], p42[[i]], p43[[i]], #xgboost
                   p44[[i]], p45[[i]], p46[[i]], p47[[i]], p48[[i]], p49[[i]], #glmnet
                   p50[[i]], p51[[i]], p52[[i]], p53[[i]], #knn
                   p54[[i]]) #kcdc
}
# view the first 6 observations in the first fold 
head(data.frame(splt[[1]]))

## 2d: calculate CV risk for each method for the ii-th validation set
# our loss function is the rank loss; so our risk is (1-AUC)
#		use the AUC() function with input as the predicted outcomes and 'labels' as the true outcomes
risk2<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2], labels=splt[[ii]][,1]))    # CV-risk for bayesglm
risk3<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,3], labels=splt[[ii]][,1]))		# CV-risk for ranger 1
risk4<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,4], labels=splt[[ii]][,1]))		# CV-risk for ranger 2
risk5<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,5], labels=splt[[ii]][,1]))		# CV-risk for ranger 3
risk6<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,6], labels=splt[[ii]][,1]))		# CV-risk for ranger 4
risk7<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,7], labels=splt[[ii]][,1]))		# CV-risk for ranger 5
risk8<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,8], labels=splt[[ii]][,1]))		# CV-risk for ranger 6
risk9<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,9], labels=splt[[ii]][,1]))		# CV-risk for ranger 7
risk10<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,10], labels=splt[[ii]][,1]))		# CV-risk for ranger 8
risk11<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,11], labels=splt[[ii]][,1]))		# CV-risk for ranger 9
risk12<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,12], labels=splt[[ii]][,1]))		# CV-risk for ranger 10
risk13<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,13], labels=splt[[ii]][,1]))		# CV-risk for ranger 11
risk14<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,14], labels=splt[[ii]][,1]))		# CV-risk for mean
risk15<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,15], labels=splt[[ii]][,1]))		# CV-risk for glm
risk16<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,16], labels=splt[[ii]][,1]))		# CV-risk for gams
risk17<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,17], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk18<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,18], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk19<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,19], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk20<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,20], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk21<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,21], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk22<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,22], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk23<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,23], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk24<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,24], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk25<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,25], labels=splt[[ii]][,1]))		# CV-risk for xgboost 
risk26<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,26], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk27<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,27], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk28<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,28], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk29<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,29], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk30<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,30], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk31<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,31], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk32<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,32], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk33<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,33], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk34<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,34], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk35<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,35], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk36<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,36], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk37<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,37], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk38<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,38], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk39<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,39], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk40<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,40], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk41<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,41], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk42<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,42], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk43<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,43], labels=splt[[ii]][,1]))		# CV-risk for xgboost
risk44<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,44], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk45<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,45], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk46<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,46], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk47<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,47], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk48<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,48], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk49<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,49], labels=splt[[ii]][,1]))		# CV-risk for glmnet
risk50<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,50], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk51<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,51], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk52<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,52], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk53<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,53], labels=splt[[ii]][,1]))		# CV-risk for KNN
risk54<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,54], labels=splt[[ii]][,1]))		# CV-risk for CDC


#----------------------------
## 3: average the estimated 5 risks across the folds to obtain 1 measure of performance for each algorithm
a<-rbind(cbind("bayesglm",mean(do.call(rbind, risk2),na.rm=T)),
         cbind("ranger1",mean(do.call(rbind, risk3),na.rm=T)),
         cbind("ranger2",mean(do.call(rbind,risk4), na.rm=T)),
         cbind("ranger3",mean(do.call(rbind,risk5), na.rm=T)),
         cbind("ranger4",mean(do.call(rbind,risk6), na.rm=T)),
         cbind("ranger5",mean(do.call(rbind,risk7), na.rm=T)),
         cbind("ranger6",mean(do.call(rbind,risk8), na.rm=T)),
         cbind("ranger7",mean(do.call(rbind,risk9), na.rm=T)),
         cbind("ranger8",mean(do.call(rbind,risk10), na.rm=T)),
         cbind("ranger9",mean(do.call(rbind,risk11), na.rm=T)),
         cbind("ranger10",mean(do.call(rbind,risk12), na.rm=T)),
         cbind("ranger11",mean(do.call(rbind,risk13), na.rm=T)),
         cbind("mean",mean(do.call(rbind,risk14), na.rm=T)),
         cbind("glm",mean(do.call(rbind,risk15), na.rm=T)),
         cbind("gam",mean(do.call(rbind,risk16), na.rm=T)),
         cbind("xgboost1",mean(do.call(rbind,risk17), na.rm=T)),
         cbind("xgboost2",mean(do.call(rbind,risk18), na.rm=T)),
         cbind("xgboost3",mean(do.call(rbind,risk19), na.rm=T)),
         cbind("xgboost4",mean(do.call(rbind,risk20), na.rm=T)),
         cbind("xgboost5",mean(do.call(rbind,risk21), na.rm=T)),
         cbind("xgboost6",mean(do.call(rbind,risk22), na.rm=T)),
         cbind("xgboost7",mean(do.call(rbind,risk23), na.rm=T)),
         cbind("xgboost8",mean(do.call(rbind,risk24), na.rm=T)),
         cbind("xgboost9",mean(do.call(rbind,risk25), na.rm=T)),
         cbind("xgboost10",mean(do.call(rbind,risk26), na.rm=T)),
         cbind("xgboost11",mean(do.call(rbind,risk27), na.rm=T)),
         cbind("xgboost12",mean(do.call(rbind,risk28), na.rm=T)),
         cbind("xgboost13",mean(do.call(rbind,risk29), na.rm=T)),
         cbind("xgboost14",mean(do.call(rbind,risk30), na.rm=T)),
         cbind("xgboost15",mean(do.call(rbind,risk31), na.rm=T)),
         cbind("xgboost16",mean(do.call(rbind,risk32), na.rm=T)),
         cbind("xgboost17",mean(do.call(rbind,risk33), na.rm=T)),
         cbind("xgboost18",mean(do.call(rbind,risk34), na.rm=T)),
         cbind("xgboost19",mean(do.call(rbind,risk35), na.rm=T)),
         cbind("xgboost20",mean(do.call(rbind,risk36), na.rm=T)),
         cbind("xgboost21",mean(do.call(rbind,risk37), na.rm=T)),
         cbind("xgboost22",mean(do.call(rbind,risk38), na.rm=T)),
         cbind("xgboost23",mean(do.call(rbind,risk39), na.rm=T)),
         cbind("xgboost24",mean(do.call(rbind,risk40), na.rm=T)),
         cbind("xgboost25",mean(do.call(rbind,risk41), na.rm=T)),
         cbind("xgboost26",mean(do.call(rbind,risk42), na.rm=T)),
         cbind("xgboost27",mean(do.call(rbind,risk43), na.rm=T)),
         cbind("glmnet1",mean(do.call(rbind,risk44), na.rm=T)),
         cbind("glmnet2",mean(do.call(rbind,risk45), na.rm=T)),
         cbind("glmnet3",mean(do.call(rbind,risk46), na.rm=T)),
         cbind("glmnet4",mean(do.call(rbind,risk47), na.rm=T)),
         cbind("glmnet5",mean(do.call(rbind,risk48), na.rm=T)),
         cbind("glmnet6",mean(do.call(rbind,risk49), na.rm=T)),
         cbind("knn1",mean(do.call(rbind,risk50), na.rm=T)),
         cbind("knn2",mean(do.call(rbind,risk51), na.rm=T)),
         cbind("knn3",mean(do.call(rbind,risk52), na.rm=T)),
         cbind("knn4",mean(do.call(rbind,risk53), na.rm=T)),
         cbind("cdc",mean(do.call(rbind,risk54), na.rm=T)))
# output a table of the CV-risk estimates
# xtable(a)
# this is one part of "SuperLearner" output. How do we get 95% CI's around these so we can plot them nicely?
saveRDS(a, "risks_momi_20200724.rds")

#----------------------------
## 4: estimate SL weights using the optim() function to minimize (1-AUC)
# X will be different. A = x 2:3 will actually be ALL the columsn of my predictions. Y = X[,1] will be observed 
#contact ashley when ready to work on this and we'll walk through 
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","bayesglm","ranger1","ranger2","ranger3","ranger4","ranger5","ranger6","ranger7","ranger8",
                                                                "ranger9","ranger10","ranger11","ranger12","mean","glm","gams","xgboost1","xgboost2","xgboost3",
                                                                "xgboost4","xgboost5","xgboost6","xgboost7","xgboost8","xgboost9","xgboost10","xgboost11","xgboost12",
                                                                "xgboost13","xgboost14","xgboost15","xgboost16","xgboost17","xgboost18","xgboost19","xgboost20","xgboost21",
                                                                "xgboost22","xgboost23","xgboost24","xgboost25","xgboost26","xgboost27","glmnet1","glmnet2","glmnet3",
                                                                "glmnet4","glmnet5","glmnet6","knn1","knn2","knn3","knn4","cdc")
head(X)

bounds = c(0, Inf)
SL.r<-function(A, y, par){
  A<-as.matrix(A)
  names(par)<-c("bayesglm","ranger1","ranger2","ranger3","ranger4","ranger5","ranger6","ranger7","ranger8",
                "ranger9","ranger10","ranger11","ranger12","mean","glm","gams","xgboost1","xgboost2","xgboost3",
                "xgboost4","xgboost5","xgboost6","xgboost7","xgboost8","xgboost9","xgboost10","xgboost11","xgboost12",
                "xgboost13","xgboost14","xgboost15","xgboost16","xgboost17","xgboost18","xgboost19","xgboost20","xgboost21",
                "xgboost22","xgboost23","xgboost24","xgboost25","xgboost26","xgboost27","glmnet1","glmnet2","glmnet3",
                "glmnet4","glmnet5","glmnet6","knn1","knn2","knn3","knn4","cdc")
  predictions <- crossprod(t(A),par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}

############################### Testing SL.r, delete when done ############################################################################################
A <- as.matrix(X[,2:55])
par <- rep(1/ncol(A), ncol(A))
names(par) <- c("bayesglm","ranger1","ranger2","ranger3","ranger4","ranger5","ranger6","ranger7","ranger8",
        "ranger9","ranger10","ranger11","ranger12","mean","glm","gams","xgboost1","xgboost2","xgboost3",
        "xgboost4","xgboost5","xgboost6","xgboost7","xgboost8","xgboost9","xgboost10","xgboost11","xgboost12",
        "xgboost13","xgboost14","xgboost15","xgboost16","xgboost17","xgboost18","xgboost19","xgboost20","xgboost21",
        "xgboost22","xgboost23","xgboost24","xgboost25","xgboost26","xgboost27","glmnet1","glmnet2","glmnet3",
        "glmnet4","glmnet5","glmnet6","knn1","knn2","knn3","knn4","cdc")
predictions <- crossprod(t(A), par)
auc_obj  <- AUC(predictions = predictions, labels = X[,1])
cvRisk <- 1-auc_obj
########################################################################################################################################################### 


init <- rep(1/54, 54)
fit <- optim(par=init, fn=SL.r, A=X[,2:55], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])
fit

alpha<-fit$par/sum(fit$par)
options(scipen=999)
alpha
sum(alpha)

alpha <- data.frame(alpha)
rownames(alpha) <- c("bayesglm","ranger1","ranger2","ranger3","ranger4","ranger5","ranger6","ranger7","ranger8",
                "ranger9","ranger10","ranger11","ranger12","mean","glm","gams","xgboost1","xgboost2","xgboost3",
                "xgboost4","xgboost5","xgboost6","xgboost7","xgboost8","xgboost9","xgboost10","xgboost11","xgboost12",
                "xgboost13","xgboost14","xgboost15","xgboost16","xgboost17","xgboost18","xgboost19","xgboost20","xgboost21",
                "xgboost22","xgboost23","xgboost24","xgboost25","xgboost26","xgboost27","glmnet1","glmnet2","glmnet3",
                "glmnet4","glmnet5","glmnet6","knn1","knn2","knn3","knn4","cdc")

write.csv(alpha, file="C:\\Users\\abc72\\Box Sync\\Dissertation backup\\Dissertation\\MOMI\\Exploratory data analysis\\alpha_test_novarselection_2020.07.24.csv")

#---------------------
## 5a: fit all algorithms to NEW data
m1<-bayesglm(formula=ch_smmtrue~.,data=D,family="binomial")
m2<-ranger(ch_smmtrue~., data=D)

## 5b: predict probabilities from each fit using all data
p1<-predict(m1,newdata=D,type="response")  # bayesglm
p2<-(predict(m2, data=D))$predictions #randomForest
predictions<-cbind(p1,p2)
head(predictions)

## 5c: for the observed data take a weighted combination of predictions using nnls coeficients as weights
y_pred <- predictions%*%alpha
p<-data.frame(y=D$ch_smmtrue,y_pred=y_pred)

## #--------------------------------------------
# verify that our work predicts similar results as SL package
a<-roc(p$y, p$y_pred, direction="auto")
C2<-data.frame(sens=a$sensitivities,spec=a$specificities)
head(C2)

###--------------------------------------------
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
