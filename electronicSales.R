##########################################################################################################
## Loading Files and Packages
##########################################################################################################
## Load libraries
library(PerformanceAnalytics)
library(Rmisc)
library(Hmisc)
library(lme4)
library(caret)
library(pROC)
library(glmertree)
library(car)
library(glmmADMB)
library(coefplot)
library(sjPlot)
library(bbmle)
library(reshape)
library(MCMCglmm)
library(mclust)
library(cluster)
library(kernlab)
library(fpc)

## Load libraries for ASREML Analyses
library(broom)
library(asreml)
library(ggplot2)
library(pascal)

## Clear Rs brain
rm(list=ls())

## Read the dataset into R
dat <- read.csv("D:/My documents/University/Job Tests/electronic Sales/ESalesdataset.csv")

##########################################################################################################
## Check the structure and shape of the data #########
## This serves several purposes:
## 1 - it shows me what kind of data I have and how it is distributed
## 2 - let's me see if there are correlations that I should keep in mind
## 3 - Let's me make some assumptions (maybe)
## 4 - Detection of possible outliers on the data
##########################################################################################################
## Number of cases on the data (might be useful for future models)
nCases<-nrow(dat)

## Check structure of the data
str(dat)
summary(dat)

## Set variables to correct format (factors or numeric)
dat$seller<-as.factor(dat$seller)

## Remove the one NA from 7d_purchases (befause I do not know what to replace it with) 
dat<-dat[complete.cases(dat$7d_purchases),,drop = T]

############### ONLY DEALING WITH PRIVATE SELLERS ###############
## Check that they are all Private Sellers
levels(dat$buyer_segment)
## I assume that all these categories refer to private sellers

##################################################################
## Take Random Sample of DataSet because the whole is too large
## Sample the sellers randomly, because you should not have half data of a seller
## for building the model and half for predicting afterwards.
nSellers<-nlevels(dat$seller)
sellerList<-levels(dat$seller)
subList<-sample(x = sellerList, size = nSellers*.001)
datSample<-dat[dat$seller %in% subList,]

## Change Sample to dat and "dat" to "origDat"
origDat<-dat
dat<-datSample

############### FACTORS ################### 
## Determine the factors
factDat<-dat[,c(1:4)]
factVars<- names(factDat)

############### QUANTITATIVE VARS ################### 
## Determine the quantitative variables
quantDat<-dat[,5:19]
quantVars<-names(quantDat)

## Check Quant variable distributions
for (i in quantVars){
  hist(dat[[i]], main = i)  
}

## Transform variables if necessary
## Let's see what errors the models report before transforming
## Let's start with scaling for now
## Scale quantitative Vars (but not Response Vars) for model fitting
predictVars<-names(dat[,5:17])
## Models require to transform the variables according to error distributions

transfVars<-names(dat[,c(6:10,12:18)])
for(i in transfVars){
  dat[[i]]<-log(dat[[i]])
}

for (i in predictVars){
  dat[paste(i,"Scaled",sep = "")]<-as.numeric(scale(dat[[i]], center = TRUE, scale = TRUE)  )
} 

## Check Quant variable distributions
for (i in quantVars){
  hist(dat[[i]], main = i)  
}


## Check that the scaled ones are there now
names(dat)
str(dat)

## Check correlations between quant variables
correlations = (rcorr(as.matrix(quantDat), type="pearson"))
chart.Correlation(quantDat)

## Drop all user levels that do not apply anymore
dat$seller<-as.factor(dat$seller)

############### RESPONSE VARS ################### 
## Create binomial response variable to know who bought or not
## This way, I can fit a classification algorithm to the data
dat$PurchaseT_F<-dat$7d_purchases  > 0

str(dat)

##########################################################################################################
## Model A - Fit Statistical Model to idenfity significant variables affecting Buy or NO BUY #########
##########################################################################################################
## Build chain of the names of the vars and NO interactions to fit the model
varList<-names(dat[,c(2:4,20:32)])

varNames = as.character()
firstVar = 0
for (i in varList){
  varNames = paste(i,varNames, sep = " + ")
}


##Try a Glm with Poisson Distrib
modelGLMER<-glmer(PurchaseT_F ~ 2d_purchasesScaled + 2d_item_viewsScaled + 2d_searchesScaled + 7d_purchasesScaled + 7d_item_viewsScaled + 7d_searchesScaled + final_price_cat_pctlScaled + final_priceScaled + 2d_bidsScaled + first_2d_bidsScaled + total_bidsScaled + start_priceScaled + auction_durationScaled + buyer_segment + 
                    (1|seller),
                  family="binomial",
                  control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000)),
                  data = dat,
                  verbose = 1)

## Check the variance
plot(modelGLMER)
## Check normality of the residuals
qqnorm(resid(modelGLMER))
## Check goodness of fit (residual deviance and degrees of freedom)
1-pchisq(sum(residuals(modelGLMER)^2),df.residual(modelGLMER))
## Check if random effects are normally distributed
sjp.glmer(modelGLMER, type = "re.qq")
## Check correlation of fixed effects
sjp.glmer(modelGLMER, type = "fe.cor")
## plot probability curve of fixed effects
sjp.glmer(modelGLMER, type = "ri.pc",
          show.se = FALSE,
          facet.grid = FALSE)
## posterior predictive simulations to test whether the model is behaving like the data in other ways
sims <- simulate(modelGLMER,nsim=1000)
nOnes <- colSums(sims==1)
par(las=1,bty="l")
plot(pt <- prop.table(table(nOnes)),
     ylab="Probability")
(obsOne <- sum(dat$Fighting==1))
points(obsOne,0.02,col="red",pch=16,cex=2)

## Since model is not behaving in a normal way, I would transform the predictive variables and try again

## Compare the models
AICtab(modelGLMER)
multiplot(modelGLMER)

summary(modelGLMER)

Anova(modelGLMER)
Anova(modelGLMER, type =3)
hist(resid(modelGLMER))

##########################################################################################################
## Test the Data ######
## Fit models with random factors because the errors of each seller are correlated
## That limits spot checking to particular algorithms that can handle Random Effects
## The models compare the results against a random Reponse 
##########################################################################################################
## Model B1 - Fit Predictive GLMER to the data #########
##########################################################################################################
## Create the K-folds to train and test the data
## Subset the data per individual because the data from each individual is correlated
## I want a model that predicts for other users, not only the ones I have
## Hence, fitting to some sellers and testing on others will do that job
folds = 5
foldSamples = createFolds(dat$seller, k = folds, list = TRUE, returnTrain = FALSE)
str(foldSamples)

## Set the Data Frame where the model metrics will be stored
glmerModelAccu = data.frame()

## Control the optimizer and number of runs to fit the glmer model
my.control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200))
## This i values is to check that the loop works properly
i = 3

for (i in 1:folds)
{
  ## Subset data into the training and testing DATA
  testData = dat[foldSamples[[i]],]
  trainData = dat[-foldSamples[[i]],]
  
  ## Model to see effects of th
  glmerModel<- glmer(PurchaseT_F~  2d_purchasesScaled + 2d_item_viewsScaled + 2d_searchesScaled + 7d_purchasesScaled + 7d_item_viewsScaled + 7d_searchesScaled + final_price_cat_pctlScaled + final_priceScaled + 2d_bidsScaled + first_2d_bidsScaled + total_bidsScaled + start_priceScaled + auction_durationScaled + buyer_segment+
                       (1|seller),
                     data = trainData, 
                     family =binomial(logit),
                     control = my.control,
                     na.action = na.exclude)
  
  ## Check predictions for training Data
  trainData$predY<-predict(object = glmerModel, newdata = trainData, type = "response")
  ## Calculate ROC for the training
  training.ROC <- roc(predictor=trainData$predY, response=as.factor(trainData$PurchaseT_F))
  ## Get the best Threshold, Shortest distance between ROC and upper left corner of graph
  bestThres<-coords(roc = training.ROC, x = "best", ret="t")
  
  ## Get the binary response for the Confusion Matrix
  trainData$predYLogistic<-trainData$predY>bestThres
  ## Obtain the Confusion Matrix
  trainConfM<-confusionMatrix(data = trainData$predYLogistic, reference = trainData$PurchaseT_F,
                              mode = "everything")
  
  ## Random Data for comparing model performance
  testData$predYRandom<-runif(nrow(testData), min=min(testData$PurchaseT_F), max=max(testData$PurchaseT_F))
  ## Calculate ROC for the Random Data
  Random.ROC <- roc(predictor=testData$predYRandom, response=as.factor(testData$PurchaseT_F))
  ## Get the binary response for the Confusion Matrix
  testData$predYLogisticR<-testData$predYRandom>bestThres
  ## Obtain the Confusion Matrix
  testConfMRandom<-confusionMatrix(data = testData$predYLogisticR, reference = testData$PurchaseT_F,
                                   mode = "everything")
  
  ## Check predictions for Testing Data
  testData$predY<-predict(object = glmerModel, newdata = testData, type = "response", allow.new.levels = TRUE)
  ## Calculate ROC for the Testing
  testing.ROC <- roc(predictor=testData$predY, response=as.factor(testData$PurchaseT_F))
  ## Get the binary response for the Confusion Matrix
  testData$predYLogistic<-testData$predY>bestThres
  ## Obtain the Confusion Matrix
  testConfM<-confusionMatrix(data = testData$predYLogistic, reference = testData$PurchaseT_F,
                             mode = "everything")
  
  ## Get the model performance values in the table (AUC and F1) for both training and testing
  tempTrainAccu = data.frame( foldNum = i, Random_AUC = Random.ROC$auc, Random_F1 = testConfMRandom$byClass["F1"],
                              Random_Accuracy = testConfMRandom$overall["Accuracy"],
                              testing_AUC = testing.ROC$auc, testing_F1 = testConfM$byClass["F1"],
                              testing_Accuracy = testConfM$overall["Accuracy"])
  ## Add to the final Table
  glmerModelAccu = rbind(glmerModelAccu,tempTrainAccu)
}

## Calcuate means of the k models for the performance measurements
sapply(glmerModelAccu, mean, na.rm=TRUE)

##########################################################################################################
## Model B2 - Fit GLMERTREE to the data #########
## 
##########################################################################################################
## Create the K-folds to train and test the data
## Subset the data per individual because the data from each individual is correlated
## I want a model that predicts for other users, not only the ones I have
## Hence, fitting to some sellers and testing on others will do that job
folds = 5
foldSamples = createFolds(dat$seller, k = folds, list = TRUE, returnTrain = FALSE)
str(foldSamples)

## Set the Data Frame where the model metrics will be stored
glmerTREEModelAccu = data.frame()

## This i values is to check that the loop works properly
i = 3


for (i in 1:folds)
{
  ## Subset data into the training and testing DATA
  testData = dat[foldSamples[[i]],]
  trainData = dat[-foldSamples[[i]],]
  
  ## Model to see effects of th
  glmerTREEModel<- glmertree(PurchaseT_F~ 2d_purchasesScaled + 2d_item_viewsScaled + 2d_searchesScaled + 7d_purchasesScaled + 7d_item_viewsScaled + 7d_searchesScaled + final_price_cat_pctlScaled + final_priceScaled + 2d_bidsScaled + first_2d_bidsScaled + total_bidsScaled + start_priceScaled + auction_durationScaled | seller | buyer_segment,
                             data = trainData, 
                             family =binomial(),
                             ranefstart = NULL, 
                             abstol = 0.001, 
                             maxit = 500, 
                             joint = TRUE, 
                             dfsplit = TRUE, 
                             verbose = TRUE, 
                             plot = FALSE,
                             na.action = na.exclude)
  
  ## Check predictions for training Data
  trainData$predY<-predict(object = glmerTREEModel, newdata = trainData, type = "response")
  ## Calculate ROC for the training
  training.ROC <- roc(predictor=trainData$predY, response=trainData$PurchaseT_F)
  
  bestThres<-coords(roc = training.ROC, x = "best", ret="t")
  
  ## Get the binary response for the Confusion Matrix
  trainData$predYLogistic<-trainData$predY>bestThres
  ## Obtain the Confusion Matrix
  trainConfM<-confusionMatrix(data = trainData$predYLogistic, reference = trainData$PurchaseT_F,
                              mode = "everything")
  
  ## Random Data for comparing model performance
  testData$predYRandom<-runif(nrow(testData), min=min(testData$PurchaseT_F), max=max(testData$PurchaseT_F))
  ## Calculate ROC for the Random Data
  Random.ROC <- roc(predictor=testData$predYRandom, response=testData$PurchaseT_F)
  ## Get the binary response for the Confusion Matrix
  testData$predYLogisticR<-testData$predYRandom>bestThres
  ## Obtain the Confusion Matrix
  testConfMRandom<-confusionMatrix(data = testData$predYLogisticR, reference = testData$PurchaseT_F,
                                   mode = "everything")
  
  ## Check predictions for Testing Data
  testData$predY<-predict(object = glmerTREEModel, newdata = testData, type = "response", allow.new.levels = TRUE)
  ## Calculate ROC for the Testing
  testing.ROC <- roc(predictor=testData$predY, response=testData$PurchaseT_F)
  ## Get the binary response for the Confusion Matrix
  testData$predYLogistic<-testData$predY>bestThres
  ## Obtain the Confusion Matrix
  testConfM<-confusionMatrix(data = testData$predYLogistic, reference = testData$PurchaseT_F,
                             mode = "everything")
  
  ## Get the model performance values in the table (AUC and F1) for both training and testing
  tempTrainAccu = data.frame( foldNum = i, Random_AUC = Random.ROC$auc, Random_F1 = testConfMRandom$byClass["F1"],
                              Random_Accuracy = testConfMRandom$overall["Accuracy"],
                              testing_AUC = testing.ROC$auc, testing_F1 = testConfM$byClass["F1"],
                              testing_Accuracy = testConfM$overall["Accuracy"])
  ## Add to the final Table
  glmerTREEModelAccu = rbind(glmerTREEModelAccu,tempTrainAccu)
}
## Calcuate means of the k models for the performance measurements
sapply(glmerTREEModelAccu, mean, na.rm=TRUE)

##########################################################################################################

##########################################################################################################
### Final Predcition model for the Binary Buy/NO Buy answer ####
##########################################################################################################
## Pick the best fitting model based on my tests and values
## Use this full model to predict if the behavior of new users

##########################################################################################################
## Model E - Create clusters #########
## Here I am making a strong assumption: The selling behaviors and traits correlate with the buying behaviors
## If that is true, then the clusters defined by the former will allow to predict the interests of the latter
## Once one of the buyers in the group buys something, we could offer the same product to the other ones
##########################################################################################################
## Determine number of clusters
## by plotting within groups sum of squares for the different Cluster numbers
wss <- (nrow(dat[,quantVars])-1)*sum(apply(dat[,quantVars],2,var))
for (i in 2:15){
  wss[i] <- sum(kmeans(dat[,quantVars],centers=i)$withinss)  
}
## Plot the sum of squares
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups SS")

# K-Means Cluster Analysis with 8 clusters based on graph
fit <- kmeans(dat[,quantVars], 8)

# get cluster means
aggregate(dat[,quantVars],by=list(fit$cluster),FUN=mean)
# append cluster assignment
mydata <- data.frame(dat, fit$cluster) 

## Plot the cluster
clusplot(dat[,quantVars], 
         fit$cluster,
         color=TRUE,
         shade=TRUE,
         labels=2,
         lines=4,
         main = "Cluster from electronic Sales dataset")


# fit model
ksvm.fit.1 = ksvm(Area~Pop*FoodQ, data=dat, kernel="rbfdot")

# summarize the fit
print(ksvm.fit.1)
