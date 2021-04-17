# This code is for the the gradient boosting techiques, also called as the sochastic gradient boosting 
# where randomess is included on top of the randomforest


#First, load all the libraries into the system
library(caret)
library(gbm)
library(doParallel)
library(ggpubr)
set.seed (1001)

#Read in your data
data <-read.csv('NMC_dataset.csv')
str(data)
#Split the data into train and test
#note: the replace option in this case is closed

data_splitting <- function (dat,splitting_ratio){
  
  n<-nrow(dat)
  n_split <- round(splitting_ratio*n)
  
  ind <-sample(n,n_split,replace = F)
  
  train <- dat[ind,]
  
  test <-dat[-ind,]
  
  return(list(train = train, test = test))
}

#Split the data according to how much you need your train and test set to be

first_split <-data_splitting(data,0.8)

#Perform the train and test splitting
train= first_split$train;test = first_split$test
#Define how you should like to train your model, for example, 10-fold cross-validation
fitControl <- trainControl(method = "cv", number =10,search = "random")


#Build your model out from all the parameters you have identified
gbmFit_IC <- train(IC ~ Li+Ni+Co+Mn+M+LC_a++LC_c+CV+CD+V_min+V_max+Mr_dopant+Mr+no_electron_M+EN_M+dif_Li_EN+dif_Ni_EN
                   +dif_E3_EN+dif_E4_EN+No_iso_dopant+E_ionisation_dopant+dif_E1_EI+dif_E2_EI+dif_E3_EI+dif_E4_EI+EA_dopant+dif_E1_EA+dif_E2_EA+dif_E3_EA+dif_E4_EA+AR_dopant+dif_E1_AR+	dif_E2_AR	
                   +dif_E3_AR+dif_E4_AR, data = train,
                 method = "gbm", 
                 trControl = fitControl,
                 tuneLength = 1000,
                 verbose = FALSE)


gbmFit_EC <- train(EC ~ Li+Ni+Co+Mn+M+LC_a++LC_c+CV+CD+V_min+V_max+Mr_dopant+Mr+no_electron_M+EN_M+dif_Li_EN+dif_Ni_EN
                   +dif_E3_EN+dif_E4_EN+No_iso_dopant+E_ionisation_dopant+dif_E1_EI+dif_E2_EI+dif_E3_EI+dif_E4_EI+EA_dopant+dif_E1_EA+dif_E2_EA+dif_E3_EA+dif_E4_EA+AR_dopant+dif_E1_AR+	dif_E2_AR	
                   +dif_E3_AR+dif_E4_AR, data = train, 
                   method = "gbm", 
                   trControl = fitControl, 
                   verbose = FALSE, 
                   tuneLength = 1000)

IC <-gbmFit_IC$bestTune
EC <- gbmFit_EC$bestTune


gbm_IC_hp_op_nmc <- gbmFit_IC$results
gbm_EC_hp_op_nmc <- gbmFit_EC$results

write.csv2(gbm_IC_hp_op_nmc,"gbm_IC_hp_op_NMC")
write.csv2(gbm_EC_hp_op_nmc,"gbm_EC_hp_op_NMC")

saveRDS(gbmFit_IC,file = "gbmFit_IC_hyperparameter_optimisation.RDS")
saveRDS(gbmFit_EC,file = "gbmFit_EC_hyperparameter_optimisation.RDS")

#See the results of all the tuning parameters you have selected
Optimisation_results_IC <- gbmFit_IC$results
Optimisation_results_EC <- gbmFit_IC$results
#Optimisation_results_V <- gbmFit_V$results

#Create a csv file for these optimisation results
write.csv(Optimisation_results_IC,"Optimisation_results_gbm_IC")
write.csv(Optimisation_results_EC,"Optimisation_results_gbm_EC")
#write.csv(Optimisation_results_V,"Optimisation_results_gbm_V")

gbmFit_IC <- readRDS("final_model_IC.rds")
a<-gbmFit_IC$bestTune
write.csv(a,'IC_optimised_hyper_gbm')

gbmFit_EC <- readRDS("final_model_EC_gbm.rds")
b<-gbmFit_EC$bestTune
write.csv(b,'EC_optimised_hyper_gbm')

#Now, make a stand alone model based on the model parameters you have selected
final_model_IC <- train(IC ~ Li+Ni+Co+Mn+M+LC_a++LC_c+CV+CD+V_min+V_max+Mr_dopant+Mr+no_electron_M+EN_M+dif_Li_EN+dif_Ni_EN
                        +dif_E3_EN+dif_E4_EN+No_iso_dopant+E_ionisation_dopant+dif_E1_EI+dif_E2_EI+dif_E3_EI+dif_E4_EI+EA_dopant+dif_E1_EA+dif_E2_EA+dif_E3_EA+dif_E4_EA+AR_dopant+dif_E1_AR+	dif_E2_AR	
                        +dif_E3_AR+dif_E4_AR, data = train,  
                        method = "gbm", 
                        trControl = fitControl, 
                        verbose = FALSE, 
                        tuneGrid = gbmFit_IC$bestTune)

final_model_EC <- train(EC ~ Li+Ni+Co+Mn+M+LC_a++LC_c+CV+CD+V_min+V_max+Mr_dopant+Mr+no_electron_M+EN_M+dif_Li_EN+dif_Ni_EN
                        +dif_E3_EN+dif_E4_EN+No_iso_dopant+E_ionisation_dopant+dif_E1_EI+dif_E2_EI+dif_E3_EI+dif_E4_EI+EA_dopant+dif_E1_EA+dif_E2_EA+dif_E3_EA+dif_E4_EA+AR_dopant+dif_E1_AR+	dif_E2_AR	
                        +dif_E3_AR+dif_E4_AR, data = train, 
                     method = "gbm", 
                     trControl = fitControl, 
                     verbose = FALSE, 
                     tuneGrid = gbmFit_EC$bestTune)

### Averaging them #####
averg <- function(data,a){
  n <- nrow(data);
  b <-  a+9;
  new_RMSE <- mean(data[a:b,1])
  new_R_square <- mean(data[a:b,2])
  return(list(new_RMSE,new_R_square))
}

a <- seq (1,nrow(IC_resample_results),by=10)

hyperparameters_gbm_ec <- data.frame(0,0)
names(hyperparameters_gbm_ec) <-c('RMSE','R_squared')

for (i in a){
  Mean_RMSE <- averg(EC_resample_results,i)[[1]]
  Mean_R_squared <- averg(EC_resample_results,i)[[2]]
  new_list_c <- data.frame(Mean_RMSE,Mean_R_squared)
  names(new_list_c) <- c('RMSE','R_squared')
  hyperparameters_gbm_ec <-rbind(hyperparameters_gbm_ec,new_list_c)
} 

hyperparameters_gbm_ic <- hyperparameters_gbm_ic[-c(1),]
hyperparameters_gbm_ec <- hyperparameters_gbm_ec[-c(1),]

write.csv(hyperparameters_gbm_ic,"train_error_gbm_ic")
write.csv(hyperparameters_gbm_ec,"train_error_gbm_ec")

#Plot the heatmap for your model
trellis.par.set(caretTheme())
plot(gbmFit_IC, metric = "RMSE", plotType = "level",
     scales = list(x = list(rot = 90)))

plot(gbmFit_EC, metric = "RMSE", plotType = "level",
     scales = list(x = list(rot = 90)))

plot(gbmFit_V, metric = "RMSE", plotType = "level",
     scales = list(x = list(rot = 90)))

#Save this model into the current working directory, under whatever the name you would like to give it
saveRDS(final_model_IC,"final_model_IC.rds")
saveRDS(final_model_EC,"final_model_EC_gbm.rds")
saveRDS(final_model_V,"final_model_V_gbm.rds")

#Plotting the variable importance graphs
summary(final_model_IC)
summary(final_model_EC)
summary(final_model_V)


#Get the resampling results
IC_resample_results <-final_model_IC$resample
EC_resample_results <-final_model_EC$resample

write.csv(IC_resample_results,"IC_resample_results")
write.csv(EC_resample_results,"EC_resample_results")

#Now test your model performance against the test set

first_split <-data_splitting(data,0.8)

#Perform the train and test splitting
train = first_split$train;test = first_split$test

final_model_IC <- train(IC ~ M+Mn+M_EN+Mr+LC_a+CD, data = train, 
                        method = "gbm", 
                        trControl = fitControl, 
                        verbose = FALSE, 
                        tuneGrid = a)

final_model_EC <- train(EC ~ M+Mn+M_EN+Mr+LC_a+CD, data = train, 
                        method = "gbm", 
                        trControl = fitControl, 
                        verbose = FALSE, 
                        tuneGrid = b)
####################   IC   ###########
predicted_value_IC <- predict(final_model_IC,newdata = test)#Start with doing the prediction 

#predicted_value_IC #Check the prediction values 

RMSE_Test_IC <- sqrt((mean((predicted_value_IC-test$IC)^2)))
fit_IC <- lm(predicted_value_IC~test$IC)
r_squared_IC <- signif(summary(fit_IC)$adj.r.squared, 5)
RMSE_Test_IC

EC_model$trainingData

final_model_IC$resample

#####################  EC    #################
predicted_value_EC <- predict(final_model_EC,newdata = test)#Start with doing the prediction 

predicted_value_EC #Check the prediction values 

RMSE_Test_EC <- sqrt((mean((predicted_value_EC-test$EC)^2)))
fit_EC <- lm(predicted_value_EC~test$EC)
fit_EC <- lm(predicted_value_EC~test$EC)
r_squared_EC <- signif(summary(fit_EC)$adj.r.squared, 5)



RMSE_Test_IC
RMSE_Test_EC
r_squared_IC
r_squared_EC

IC_5 <- cbind(predicted_value_IC,test$IC)
EC_4 <- cbind(predicted_value_EC,test$EC)

write.csv(IC_5,"gbm_ic_pred_versus_test.csv")
write.csv(EC_4,"gbm_ec_pred_versus_test.csv")

write.csv(final_model_IC$resample,"train_error_gbm_ic.csv")
write.csv(final_model_EC$resample,"train_error_gbm_ec.csv")


fit_IC <- lm(predicted_value_IC ~test$IC)
fit_EC <- lm(predicted_value_EC~test$EC)

print(ggplotRegression((fit_IC))+labs(y="Predicted Initial Capacity (mAh/g)",x="Experimental Initial Capacity(mAh/g)" ))
print(ggplotRegression((fit_EC))+labs(y="Predicted End Capacity (mAh/g)",x="Experimental End Capacity(mAh/g)" ))


#################### Plot the final results graphs ###############

ggplotRegression <- function (fit) {
  
  ggplot(fit$model, aes_string(x = names(fit$model)[2], y = names(fit$model)[1])) + 
    geom_point(color = "blue",size=4) +
    theme_bw()+
    stat_smooth(method = "lm", col = "red") +
    labs(title = paste("Adj R2 = ",signif(summary(fit)$adj.r.squared, 5),
                       "Intercept =",signif(fit$coef[[1]],5 ),
                       " Slope =",signif(fit$coef[[2]], 5),
                       " P =",signif(summary(fit)$coef[2,4], 5)))+ 
    theme(text = element_text(size = 25),axis.text.x = element_text(hjust = 1),
          axis.title = element_text(size = rel(1),face = "bold"),
          axis.text =  element_text(size = rel(1.5),face = "bold"),plot.background = element_rect(fill = "white",
                                                                                                  colour = "white"))
}

fit_IC <- lm(predicted_value_IC ~test$IC)
fit_EC <- lm(predicted_value_EC~test$EC)

print(ggplotRegression((fit_IC))+labs(y="Predicted Initial Capacity (mAh/g)",x="Experimental Initial Capacity(mAh/g)" ))
print(ggplotRegression((fit_EC))+labs(y="Predicted End Capacity (mAh/g)",x="Experimental End Capacity(mAh/g)" ))


################################################

#Now we are going to plot the variable importance graphs from the optimal model obtained above

Imp_IC <- varImp(final_model_IC,scale =T)
Imp_IC

Imp_EC <- varImp(final_model_EC,scale =T)
Imp_EC

Imp_V <- varImp(final_model_V,scale =T)
Imp_V
plot(Imp_IC)
plot(Imp_EC)
plot(Imp_V)

#Caculate the RMSE values between the observed values and predicted values

#Plot the results out by using the observed value agian the predicted value
#As well as plotting a linear line to show the possible linear correlation in between the data. 

ggplotRegression <- function (fit) {
  
  ggplot(fit$model, aes_string(x = names(fit$model)[2], y = names(fit$model)[1])) + 
    geom_point(color = "blue",size=4) +
    theme_bw()+
    stat_smooth(method = "lm", col = "red") +
    labs(title = paste("Adj R2 = ",signif(summary(fit)$adj.r.squared, 5),
                       "Intercept =",signif(fit$coef[[1]],5 ),
                       " Slope =",signif(fit$coef[[2]], 5),
                       " P =",signif(summary(fit)$coef[2,4], 5)))+ 
    theme(text = element_text(size = 25),axis.text.x = element_text(hjust = 1),
          axis.title = element_text(size = rel(1),face = "bold"),
          axis.text =  element_text(size = rel(1.5),face = "bold"),plot.background = element_rect(fill = "white",
                                                                                                  colour = "white"))
}

fit_IC <- lm(predict_y_ic_test ~test$IC)
fit_EC <- lm(predict_y_ec_test~test$EC)

print(ggplotRegression((fit_IC))+labs(y="Predicted Initial Capacity (mAh/g)",x="Experimental Initial Capacity(mAh/g)" ))
print(ggplotRegression((fit_EC))+labs(y="Predicted End Capacity (mAh/g)",x="Experimental End Capacity(mAh/g)" ))
