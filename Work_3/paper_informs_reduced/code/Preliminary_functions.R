## Split data into train and test 
tr_test_st <- function(data_in)
{
  smp_size <- floor(0.75 * nrow(data_in))
  # set.seed(12345)
  set.seed(1234)
  train_ind <- sample(seq_len(nrow(data_in)), size = smp_size)
  train <- data_in[train_ind, ]
  test <- data_in[-train_ind, ]
  return (list(train,test))
}


# impute_manual <- function(data_in)
# {
  # for(j in 1:ncol(data_in))
  # {
    # if(is.factor(data_in[,j]))
    # {
     # data_in[is.na(data_in[,j]),j] <- names(sort(-table(data_in[,j])))[1] 
	 # # data_in[is.na(data_in[,j]),j] <- names(sort(table(data_in[,j])))[1]
    # }
    # else
    # {
      # data_in[is.na(data_in[,j]),j]  <- mean(data_in[,j],na.rm=T)
    # }  
  # }
  # return(data_in)
# }
## Better function is to use hotdeck of VIM for mixed data:
impute_manual <- function(data_in)
{
return(hotdeck(data_in, imp_var=FALSE))
}


impute_test <- function(data_tr,data_test)
{
#   for(j in 1:(ncol(data_test)) )  ## Two extra columns added to data_test as compared to data_train: (?)
  for(j in names(data_tr))  
  {
    if(is.factor(data_test[,j]))
    {
      data_test[is.na(data_test[,j]),j] <- names(sort(-table(data_tr[,j])))[1]
	  # data_test[is.na(data_test[,j]),j] <- names(sort(table(data_tr[,j])))[1]
    }
    else
    {
      data_test[is.na(data_test[,j]),j]  <- mean(data_tr[,j],na.rm=T)
    }  
  }
  return(data_test)
}

## Missing proportion in training data:
miss_prop <- function(data_in)
{
  miss_prop_cols <- sapply(data_in, function(y) sum(length(which(is.na(y)))))
  miss_proportion <- miss_prop_cols/(dim(data_in)[1])*100
  return(miss_proportion)
}

## 
missing_percentage <- function(data_in)
{
  data_present <- data.frame( matrix(as.integer(!is.na(data_in)),nrow = nrow(data_in), ncol = ncol(data_in))) 
  names(data_present) <- names(data_in) 
  missing_vals <- nrow(data_present) - colSums(data_present)
  missingness <- sum(missing_vals)/(dim(data_present)[1]*dim(data_present)[2])*100
  return(missingness)  
}


model_fit_comparisons_reg <- function(models)
{
  return(lapply(models, function(x) summary(x)$r.squared))
}

# https://thestatsgeek.com/2014/02/08/r-squared-in-logistic-regression/
model_fit_comparisons_logitreg <- function(models)
{
  return(lapply(models, function(x) 
  {
    y_ = x$y
    nullmod <- glm(y_ ~ 1, family="binomial")
    denom <- logLik(nullmod)
    1-logLik(x)/denom
  }
  ))
}

feature_drop_loss <- function(subsets,X_in,Y_in, model, models,cluster_centers_subsets)
{
  data_train <- cbind(X_in, Y_in)
  pred_insample <- pred_reg(models, data_train,X_in,Y_in,cluster_centers_subsets)
  pred_insample_rmse <- rmsec(unlist(Y_in),pred_insample)
  feature_imp <- vector(length = ncol(X_in))

  for(feature in 1:ncol(X_in))
  {
    subsets_modified <- lapply(subsets, function(sub) 
    {
      sub[,names(sub) == names(X_in)[feature]] <- runif(n= nrow(sub))
      return(sub)
    }
    )
    models_modified <- lapply(subsets_modified,model)
    pred_insample_modified <- pred_reg(models_modified, data_train,X_in,Y_in,cluster_centers_subsets)
    pred_insample_rmse_modified <- rmsec(unlist(Y_in),pred_insample_modified)
    feature_imp[feature] =  pred_insample_rmse_modified - pred_insample_rmse
  }
  feature_imp_matrix <- data.frame(variable = names(X_in),delta_rmse = feature_imp)
  return(feature_imp_matrix)
}

mse_reg <- function(model) { mean((model$residuals)^2) } ##NN
mse_gbm <- function(model) {model$train.error[length(model$train.error)]} ##NN


mse_rf <- function(model) {model$train.error[length(model$train.error)]} 

e_logit <- function(model) {summary(model)$deviance }

acc_out <- function( y_predict, y_actual)
{ 
  if(class(y_predict)=='factor')
  {
    y_predict = as.integer(as.character(y_predict))
  }
  
  ## Convert odds ratio to class values
  y_predict = ifelse(y_predict >= 0.5, 1, 0)
  
  acc_ = sum(as.integer(y_actual == y_predict))/length(y_actual) 
  ## accuracy(y_actual,y_predict)
  auc_ = auc(y_actual,y_predict)
  recall_ = sum(as.integer(y_actual == 1 & y_predict == 1))/sum(as.integer(y_actual ==1)) 
  ## recall(data = y_predict, reference = y_actual, relevant = ref)  ## From caret
  precision_ =  sum(as.integer(y_actual == 1 & y_predict == 1))/sum(as.integer(y_predict ==1)) 
  ## precision(data = y_predict, reference = y_actual, relevant = ref)
  f1_ =  2*(recall_ * precision_)/(recall_ + precision_ )
  ## F_meas(data = y_predict, reference = y_actual, relevant = ref)
  pred_ <- c(round(acc_,4), round(auc_,4) , round(recall_,4) , round(precision_,4) ,round(f1_,4) )
  return(pred_)
}


## Model errors of subsets
maec <- function(actual,predicted)
{ 
try( mean(as.matrix(abs(actual-predicted))) )
 # mean(as.matrix(abs(actual-predicted)), na.rm=T) 
}

rmsec <- function(actual,predicted)
{ 
  try( sqrt(mean((actual-predicted)^2)) )  
 # sqrt(mean((actual-predicted)^2, na.rm=T))  
}

mapec <- function(actual,predicted)
{ 
try (mean(as.matrix(abs(actual-predicted)/actual))*100 )
 # mean(as.matrix(abs(actual-predicted)/actual), na.rm=T)*100 
}

smapec <- function(actual,predicted)
{ 
try ( mean(as.matrix(2*abs(actual-predicted)/( abs(actual)+abs(predicted) )) )*100  )
 # mean(as.matrix(2*abs(actual-predicted)/( abs(actual)+abs(predicted) )) , na.rm=T)*100 
}


predperf <- function(y_predict, y_actual,count_ind=0)
{
if(count_ind == 1)
 {
  y_predict <- round(y_predict) 
 }  
  perf <- c(rmsec(y_actual,y_predict), maec(y_actual,y_predict), smapec(y_actual,y_predict), mapec(y_actual,y_predict))
  return(perf)    
}

pred_fun <- function(model, newdata, type="class")
{
    if( any(class(model) == "numeric" )){
      predictions <- rep(0, nrow(newdata))
	} else if(any(class(model) == "glm") || any(class(model) == "negbin" ) || any(class(model) == "earth" )){
      predictions <- predict(model,newdata = newdata,type = "response")
    }  else if(any(class(model) == "gbm")){
      predictions <- predict(model,newdata = newdata,type = "response", n.trees=model$n.trees)
	}  else if(any(class(model) == "ranger")){
      predictions <- predict(model,newdata)$predictions
    } else if(any(class(model) == "rpart")){
	    predictions <- predict(model,newdata = newdata)
	   if(type == "anova") { return(predictions) }
       else{  return(predictions[,2] ) }
      
    } else if (any(class(model) == "glmnet")){
	     X_p = newdata[,!names(newdata)%in% names(data_train_y)]
	     predictions <- predict(model, newx = X_p )
	} else if (any(class(model) == "xgb.Booster")){
	newdata = as.data.frame(newdata)
	X_p = data.matrix(newdata[,!names(newdata)%in% names(data_train_y)])
	X_p <- xgb.DMatrix(X_p)
	     predictions <- predict(model, newdata = X_p )
	} else if (any(class(model) == "keras.engine.training.Model" )){
	  X_p = data.matrix(newdata[,!names(newdata)%in% names(data_train_y)])
	  ## The predict function of keras does not accept a single tuple, so we do the below trick
	  X_temp = rbind(X_p, X_p)
	     predictions <- predict(model, X_temp)[1]  
	} else    {
      predictions <- predict(model, newdata)
    }
   	#     if(class(model) == "lm" || class(model) == "classbag" ||  class(model) == "randomForest" )
	return(predictions)
}



