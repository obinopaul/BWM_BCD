## Note: We have commented lazy decision trees benchmark in
### the reproducibility run as it takes more than 12 hours to ### run the module whereas codeocean has limite resources. 

gen_missing_adult <- function(adult,propMissing)
{  
  set.seed(1234)
  miss_1 <- sample(nrow(adult),propMissing*nrow(adult),replace=FALSE )
  set.seed(111)
  miss_2 <- sample(nrow(adult),propMissing*nrow(adult),replace=FALSE )
  
  adult_mis <- adult  
  
  adult_mis[miss_1,1:3] <- NA
  adult_mis[miss_2,6:8] <- NA

  for(col in 1:(ncol(adult_mis)-1))
  {
    m <- 12 + col  ##For reproducibility
    set.seed(m)
    miss_temp <- sample(nrow(adult_mis),nrow(adult_mis)*0.02,replace=FALSE )
    adult_mis[miss_temp,col] <- NA
  }
  
  return(adult_mis)
}

reg_train <- function(data_in)
{
  X <- data_in[,names(data_in) %in% names(data_train_x) ]
  Y <- data_in[,names(data_in) %in% names(data_train_y) ]
  datam <- data.table(X,Y)
  fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))
  # reg <- stepAIC(glm( fmla, data=data_in,family="poisson"),direction="both",trace=FALSE)
  reg <- glm( fmla, data=datam,family="binomial")
  
  return(reg)
}

tree_train <- function(data_in)
{
  X <- data_in[,names(data_in) %in% names(data_train_x) ]
  Y <- data_in[,names(data_in) %in% names(data_train_y) ]
  datam <- data.table(X,Y) 
  fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))
  #reg <- gbm( fmla, data=datam ,verbose=FALSE,n.trees = 500,shrinkage=0.1,interaction.depth = 3)
  reg <- gbm( fmla, data=datam, distribution = 'bernoulli', verbose=FALSE,n.trees = 500)
  ### Note that the output should not be factor, for gbm model!!
  return(reg)
}

## Get the data and do some pre-processing
data_all <- read.csv("/root/capsule/data/adult.csv")
adult = data_all
adult <- as.data.frame(unclass(adult),stringsAsFactors=TRUE)
## adult = maxlevels(adult, maxlevels = 5, na.omit = FALSE)

##  https://stackoverflow.com/questions/15533594/r-factor-levels-recode-rest-to-other
## INstead of throwing away factor variables with high number of levels, we group infrequent levels to 'Other' as follows: 

recodeLevels <- function (ds, mp = 0.10, ml = 5) {
  var_list <- names(ds[sapply(ds, is.factor)])
  # remove less frequent levels in factor
  n <- nrow(ds)
  # keep levels with more then mp percent of cases
  for (i in var_list){
    keep <- levels(ds[[i]])[table(ds[[i]]) > mp * n]
    levels(ds[[i]])[which(!levels(ds[[i]])%in%keep)] <- "other"
  }
  # keep top ml levels
  for (i in var_list){
    keep <- names(sort(table(ds[i]),decreasing=TRUE)[1:ml])
    levels(ds[[i]])[which(!levels(ds[[i]])%in%keep)] <- "other"
  }
  return(ds)
}

#adult$workclass[adult$workclass==' ?'] = NA
adult$salary = ifelse(adult$salary == '>=50k', 1, 0)
# adult$salary <- factor(adult$salary)
adult <- adult[,!names(adult) %in% c('fnlwt',"capital.gain" ,  "capital.loss") ]
adult <- recodeLevels(adult)

dat_list <- list()
dat_list[[1]] <- gen_missing_adult(adult,0.10) 
dat_list[[2]] <- gen_missing_adult(adult,0.20)
dat_list[[3]] <- gen_missing_adult(adult,0.30) 
dat_list[[4]] <- gen_missing_adult(adult,0.40)
dat_list[[5]] <- gen_missing_adult(adult,0.50) 
dat_list[[6]] <- gen_missing_adult(adult,0.60)
dat_list[[7]] <- gen_missing_adult(adult,0.70)
dat_list[[8]] <- gen_missing_adult(adult,0.80) 
dat_list[[9]] <- gen_missing_adult(adult,0.90) 
### 70% missing:
data_train <- tr_test_st(dat_list[[3]])[[1]]
data_test <- tr_test_st(dat_list[[3]])[[2]]

# sapply(dat_list[[7]], function(y) sum(length(which(is.na(y)))))

####
## Change this for each dataset
####
outcome = "salary"
###
###

data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)

data_train_no_miss <- tr_test_st(adult)[[1]]
data_test_no_miss <- tr_test_st(adult)[[2]]

p_brm_reg <- matrix(NA, nrow = 9, ncol = 5)
p_brm_gbm <- matrix(NA, nrow = 9, ncol = 5)
p_brm_n_ov_reg <- matrix(NA, nrow = 9, ncol = 5)
p_brm_n_ov_gbm <- matrix(NA, nrow = 9, ncol = 5)
p_listwise_reg <- matrix(NA, nrow = 9, ncol = 5)
p_listwise_gbm <- matrix(NA, nrow = 9, ncol = 5)
p_columnwise_reg <- matrix(NA, nrow = 9, ncol = 5)
p_columnwise_gbm <- matrix(NA, nrow = 9, ncol = 5)
p_si_reg <- matrix(NA, nrow = 9, ncol = 5)
p_si_gbm <- matrix(NA, nrow = 9, ncol = 5)
p_mi_reg <- matrix(NA, nrow = 9, ncol = 5)
p_mi_gbm <- matrix(NA, nrow = 9, ncol = 5)
p_cart <- matrix(NA, nrow = 9, ncol = 5)
p_imsf <- matrix(NA, nrow = 9, ncol = 5)
p_discom <- matrix(NA, nrow = 9, ncol = 5)
p_refe_reg <- matrix(NA, nrow = 9, ncol = 5)
p_refe_gbm <- matrix(NA, nrow = 9, ncol = 5)
p_gain_reg <- matrix(NA, nrow = 9, ncol = 5)
p_gain_gbm <- matrix(NA, nrow = 9, ncol = 5)

p_cart_i <- matrix(NA, nrow = 9, ncol = 5)
p_lazy_reg <- matrix(NA, nrow = 9, ncol = 5)
p_lazy_tree <- matrix(NA, nrow = 9, ncol = 5)


brm_pred <- list()
brm_n_ov_pred <- list()
listwise_pred <- list()
columnwise_pred <- list()
mi_pred <- list()
si_pred <- list()
cart_pred <- list()
imsf_pred <- list()
discom_pred <- list()
refe_pred <- list()
gain_pred <- list()

cart_pred_i <- list()
lazy_pred <- list()


start=proc.time()

data_train <- tr_test_st(dat_list[[1]])[[1]]
data_test <- tr_test_st(dat_list[[1]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)

try({
brm_pred[[1]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[1,] <- acc_out(brm_pred[[1]][[1]], Y_outcome)
p_brm_gbm[1,] <- acc_out(brm_pred[[1]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[1]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[1,] <- acc_out(brm_n_ov_pred[[1]][[1]], Y_outcome)
p_brm_n_ov_gbm[1,] <- acc_out(brm_n_ov_pred[[1]][[2]], Y_outcome)
})

try({
listwise_pred[[1]] <- listwise_model(data_train, data_test)
p_listwise_reg[1,] <- acc_out(listwise_pred[[1]][[1]], Y_outcome)
p_listwise_gbm[1,] <- acc_out(listwise_pred[[1]][[2]], Y_outcome)
})

try({
columnwise_pred[[1]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[1,] <- acc_out(columnwise_pred[[1]][[1]], Y_outcome)
p_columnwise_gbm[1,] <- acc_out(columnwise_pred[[1]][[2]], Y_outcome)
})

try({
mi_pred[[1]] <- mi_model(data_train, data_test)
p_mi_reg[1,] <- acc_out(mi_pred[[1]][[1]], Y_outcome)
p_mi_gbm[1,] <- acc_out(mi_pred[[1]][[2]], Y_outcome)
})

try({
si_pred[[1]] <- si_model(data_train, data_test)
p_si_reg[1,] <- acc_out(si_pred[[1]][[1]], Y_outcome)
p_si_gbm[1,] <- acc_out(si_pred[[1]][[2]], Y_outcome)
})

try({
cart_pred[[1]] <- cart_model(data_train, data_test)
p_cart[1,] <- acc_out(cart_pred[[1]], Y_outcome)
})

try({
imsf_pred[[1]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[1,] <- acc_out(imsf_pred[[1]], Y_outcome)
})

try({
discom_pred[[1]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[1,] <- acc_out(discom_pred[[1]], Y_outcome)
})

try({
refe_pred[[1]] <- refe_model(data_train, data_test)
p_refe_reg[1,] <- acc_out(refe_pred[[1]][[1]], Y_outcome)
p_refe_gbm[1,] <- acc_out(refe_pred[[1]][[2]], Y_outcome)
})

try({
gain_pred[[1]] <- gain_model(data_train, data_test)
p_gain_reg[1,] <- acc_out(gain_pred[[1]][[1]], Y_outcome)
p_gain_gbm[1,] <- acc_out(gain_pred[[1]][[2]], Y_outcome)
})

try({
cart_pred_i[[1]] <- cart_i_model(data_train, data_test)
p_cart_i[1,] <- acc_out(cart_pred_i[[1]], Y_outcome)
})

try({
#lazy_pred[[1]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[1,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[1,] <- acc_out(Y_outcome, Y_outcome)
})

#### 2 ####

data_train <- tr_test_st(dat_list[[2]])[[1]]
data_test <- tr_test_st(dat_list[[2]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)

try({
brm_pred[[2]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[2,] <- acc_out(brm_pred[[2]][[1]], Y_outcome)
p_brm_gbm[2,] <- acc_out(brm_pred[[2]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[2]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[2,] <- acc_out(brm_n_ov_pred[[2]][[1]], Y_outcome)
p_brm_n_ov_gbm[2,] <- acc_out(brm_n_ov_pred[[2]][[2]], Y_outcome)
})

try({
listwise_pred[[2]] <- listwise_model(data_train, data_test)
p_listwise_reg[2,] <- acc_out(listwise_pred[[2]][[1]], Y_outcome)
p_listwise_gbm[2,] <- acc_out(listwise_pred[[2]][[2]], Y_outcome)
})

try({
columnwise_pred[[2]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[2,] <- acc_out(columnwise_pred[[2]][[1]], Y_outcome)
p_columnwise_gbm[2,] <- acc_out(columnwise_pred[[2]][[2]], Y_outcome)
})

try({
mi_pred[[2]] <- mi_model(data_train, data_test)
p_mi_reg[2,] <- acc_out(mi_pred[[2]][[1]], Y_outcome)
p_mi_gbm[2,] <- acc_out(mi_pred[[2]][[2]], Y_outcome)
})

try({
si_pred[[2]] <- si_model(data_train, data_test)
p_si_reg[2,] <- acc_out(si_pred[[2]][[1]], Y_outcome)
p_si_gbm[2,] <- acc_out(si_pred[[2]][[2]], Y_outcome)
})

try({
cart_pred[[2]] <- cart_model(data_train, data_test)
p_cart[2,] <- acc_out(cart_pred[[2]], Y_outcome)
})

try({
imsf_pred[[2]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[2,] <- acc_out(imsf_pred[[2]], Y_outcome)
})

try({
discom_pred[[2]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[2,] <- acc_out(discom_pred[[2]], Y_outcome)
})

try({
refe_pred[[2]] <- refe_model(data_train, data_test)
p_refe_reg[2,] <- acc_out(refe_pred[[2]][[1]], Y_outcome)
p_refe_gbm[2,] <- acc_out(refe_pred[[2]][[2]], Y_outcome)
})

try({
gain_pred[[2]] <- gain_model(data_train, data_test)
p_gain_reg[2,] <- acc_out(gain_pred[[2]][[1]], Y_outcome)
p_gain_gbm[2,] <- acc_out(gain_pred[[2]][[2]], Y_outcome)
})

try({
cart_pred_i[[2]] <- cart_i_model(data_train, data_test)
p_cart_i[2,] <- acc_out(cart_pred_i[[2]], Y_outcome)
})

try({
# lazy_pred[[2]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[2,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[2,] <- acc_out(Y_outcome, Y_outcome)
})



#### subset 3

data_train <- tr_test_st(dat_list[[3]])[[1]]
data_test <- tr_test_st(dat_list[[3]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)

try({
brm_pred[[3]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[3,] <- acc_out(brm_pred[[3]][[1]], Y_outcome)
p_brm_gbm[3,] <- acc_out(brm_pred[[3]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[3]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[3,] <- acc_out(brm_n_ov_pred[[3]][[1]], Y_outcome)
p_brm_n_ov_gbm[3,] <- acc_out(brm_n_ov_pred[[3]][[2]], Y_outcome)
})

try({
listwise_pred[[3]] <- listwise_model(data_train, data_test)
p_listwise_reg[3,] <- acc_out(listwise_pred[[3]][[1]], Y_outcome)
p_listwise_gbm[3,] <- acc_out(listwise_pred[[3]][[2]], Y_outcome)
})

try({
columnwise_pred[[3]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[3,] <- acc_out(columnwise_pred[[3]][[1]], Y_outcome)
p_columnwise_gbm[3,] <- acc_out(columnwise_pred[[3]][[2]], Y_outcome)
})

try({
mi_pred[[3]] <- mi_model(data_train, data_test)
p_mi_reg[3,] <- acc_out(mi_pred[[3]][[1]], Y_outcome)
p_mi_gbm[3,] <- acc_out(mi_pred[[3]][[2]], Y_outcome)
})

try({
si_pred[[3]] <- si_model(data_train, data_test)
p_si_reg[3,] <- acc_out(si_pred[[3]][[1]], Y_outcome)
p_si_gbm[3,] <- acc_out(si_pred[[3]][[2]], Y_outcome)
})

try({
cart_pred[[3]] <- cart_model(data_train, data_test)
p_cart[3,] <- acc_out(cart_pred[[3]], Y_outcome)
})

try({
imsf_pred[[3]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[3,] <- acc_out(imsf_pred[[3]], Y_outcome)
})

try({
discom_pred[[3]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[3,] <- acc_out(discom_pred[[3]], Y_outcome)
})

try({
refe_pred[[3]] <- refe_model(data_train, data_test)
p_refe_reg[3,] <- acc_out(refe_pred[[3]][[1]], Y_outcome)
p_refe_gbm[3,] <- acc_out(refe_pred[[3]][[2]], Y_outcome)
})

try({
gain_pred[[3]] <- gain_model(data_train, data_test)
p_gain_reg[3,] <- acc_out(gain_pred[[3]][[1]], Y_outcome)
p_gain_gbm[3,] <- acc_out(gain_pred[[3]][[2]], Y_outcome)
})

try({
cart_pred_i[[3]] <- cart_i_model(data_train, data_test)
p_cart_i[3,] <- acc_out(cart_pred_i[[3]], Y_outcome)
})

try({
#lazy_pred[[3]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[3,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[3,] <- acc_out(Y_outcome, Y_outcome)
})


#### subset 4

data_train <- tr_test_st(dat_list[[4]])[[1]]
data_test <- tr_test_st(dat_list[[4]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)


try({
brm_pred[[4]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[4,] <- acc_out(brm_pred[[4]][[1]], Y_outcome)
p_brm_gbm[4,] <- acc_out(brm_pred[[4]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[4]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[4,] <- acc_out(brm_n_ov_pred[[4]][[1]], Y_outcome)
p_brm_n_ov_gbm[4,] <- acc_out(brm_n_ov_pred[[4]][[2]], Y_outcome)
})

try({
listwise_pred[[4]] <- listwise_model(data_train, data_test)
p_listwise_reg[4,] <- acc_out(listwise_pred[[4]][[1]], Y_outcome)
p_listwise_gbm[4,] <- acc_out(listwise_pred[[4]][[2]], Y_outcome)
})

try({
columnwise_pred[[4]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[4,] <- acc_out(columnwise_pred[[4]][[1]], Y_outcome)
p_columnwise_gbm[4,] <- acc_out(columnwise_pred[[4]][[2]], Y_outcome)
})

try({
mi_pred[[4]] <- mi_model(data_train, data_test)
p_mi_reg[4,] <- acc_out(mi_pred[[4]][[1]], Y_outcome)
p_mi_gbm[4,] <- acc_out(mi_pred[[4]][[2]], Y_outcome)
})

try({
si_pred[[4]] <- si_model(data_train, data_test)
p_si_reg[4,] <- acc_out(si_pred[[4]][[1]], Y_outcome)
p_si_gbm[4,] <- acc_out(si_pred[[4]][[2]], Y_outcome)
})

try({
cart_pred[[4]] <- cart_model(data_train, data_test)
p_cart[4,] <- acc_out(cart_pred[[4]], Y_outcome)
})

try({
imsf_pred[[4]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[4,] <- acc_out(imsf_pred[[4]], Y_outcome)
})

try({
 discom_pred[[4]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[4,] <- acc_out(discom_pred[[4]], Y_outcome)
})

try({
refe_pred[[4]] <- refe_model(data_train, data_test)
p_refe_reg[4,] <- acc_out(refe_pred[[4]][[1]], Y_outcome)
p_refe_gbm[4,] <- acc_out(refe_pred[[4]][[2]], Y_outcome)
})

try({
gain_pred[[4]] <- gain_model(data_train, data_test)
p_gain_reg[4,] <- acc_out(gain_pred[[4]][[1]], Y_outcome)
p_gain_gbm[4,] <- acc_out(gain_pred[[4]][[2]], Y_outcome)
})

try({
cart_pred_i[[4]] <- cart_i_model(data_train, data_test)
p_cart_i[4,] <- acc_out(cart_pred_i[[4]], Y_outcome)
})

try({
#lazy_pred[[4]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[4,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[4,] <- acc_out(Y_outcome, Y_outcome)
})


##
#### subset 5

data_train <- tr_test_st(dat_list[[5]])[[1]]
data_test <- tr_test_st(dat_list[[5]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)


try({
brm_pred[[5]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[5,] <- acc_out(brm_pred[[5]][[1]], Y_outcome)
p_brm_gbm[5,] <- acc_out(brm_pred[[5]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[5]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[5,] <- acc_out(brm_n_ov_pred[[5]][[1]], Y_outcome)
p_brm_n_ov_gbm[5,] <- acc_out(brm_n_ov_pred[[5]][[2]], Y_outcome)
})

try({
listwise_pred[[5]] <- listwise_model(data_train, data_test)
p_listwise_reg[5,] <- acc_out(listwise_pred[[5]][[1]], Y_outcome)
p_listwise_gbm[5,] <- acc_out(listwise_pred[[5]][[2]], Y_outcome)
})

try({
columnwise_pred[[5]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[5,] <- acc_out(columnwise_pred[[5]][[1]], Y_outcome)
p_columnwise_gbm[5,] <- acc_out(columnwise_pred[[5]][[2]], Y_outcome)
})

try({
mi_pred[[5]] <- mi_model(data_train, data_test)
p_mi_reg[5,] <- acc_out(mi_pred[[5]][[1]], Y_outcome)
p_mi_gbm[5,] <- acc_out(mi_pred[[5]][[2]], Y_outcome)
})

try({
si_pred[[5]] <- si_model(data_train, data_test)
p_si_reg[5,] <- acc_out(si_pred[[5]][[1]], Y_outcome)
p_si_gbm[5,] <- acc_out(si_pred[[5]][[2]], Y_outcome)
})

try({
cart_pred[[5]] <- cart_model(data_train, data_test)
p_cart[5,] <- acc_out(cart_pred[[5]], Y_outcome)
})

try({
imsf_pred[[5]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[5,] <- acc_out(imsf_pred[[5]], Y_outcome)
})

try({
 discom_pred[[5]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[5,] <- acc_out(discom_pred[[5]], Y_outcome)
})

try({
refe_pred[[5]] <- refe_model(data_train, data_test)
p_refe_reg[5,] <- acc_out(refe_pred[[5]][[1]], Y_outcome)
p_refe_gbm[5,] <- acc_out(refe_pred[[5]][[2]], Y_outcome)
})

try({
gain_pred[[5]] <- gain_model(data_train, data_test)
p_gain_reg[5,] <- acc_out(gain_pred[[5]][[1]], Y_outcome)
p_gain_gbm[5,] <- acc_out(gain_pred[[5]][[2]], Y_outcome)
})

try({
cart_pred_i[[5]] <- cart_i_model(data_train, data_test)
p_cart_i[5,] <- acc_out(cart_pred_i[[5]], Y_outcome)
})

try({
#lazy_pred[[5]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[5,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[5,] <- acc_out(Y_outcome, Y_outcome)
})



### 
#### subset 6

data_train <- tr_test_st(dat_list[[6]])[[1]]
data_test <- tr_test_st(dat_list[[6]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)

try({
brm_pred[[6]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[6,] <- acc_out(brm_pred[[6]][[1]], Y_outcome)
p_brm_gbm[6,] <- acc_out(brm_pred[[6]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[6]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[6,] <- acc_out(brm_n_ov_pred[[6]][[1]], Y_outcome)
p_brm_n_ov_gbm[6,] <- acc_out(brm_n_ov_pred[[6]][[2]], Y_outcome)
})

try({
listwise_pred[[6]] <- listwise_model(data_train, data_test)
p_listwise_reg[6,] <- acc_out(listwise_pred[[6]][[1]], Y_outcome)
p_listwise_gbm[6,] <- acc_out(listwise_pred[[6]][[2]], Y_outcome)
})

try({
columnwise_pred[[6]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[6,] <- acc_out(columnwise_pred[[6]][[1]], Y_outcome)
p_columnwise_gbm[6,] <- acc_out(columnwise_pred[[6]][[2]], Y_outcome)
})

try({
mi_pred[[6]] <- mi_model(data_train, data_test)
p_mi_reg[6,] <- acc_out(mi_pred[[6]][[1]], Y_outcome)
p_mi_gbm[6,] <- acc_out(mi_pred[[6]][[2]], Y_outcome)
})

try({
si_pred[[6]] <- si_model(data_train, data_test)
p_si_reg[6,] <- acc_out(si_pred[[6]][[1]], Y_outcome)
p_si_gbm[6,] <- acc_out(si_pred[[6]][[2]], Y_outcome)
})

try({
cart_pred[[6]] <- cart_model(data_train, data_test)
p_cart[6,] <- acc_out(cart_pred[[6]], Y_outcome)
})

try({
imsf_pred[[6]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[6,] <- acc_out(imsf_pred[[6]], Y_outcome)
})

try({
discom_pred[[6]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[6,] <- acc_out(discom_pred[[6]], Y_outcome)
})

try({
refe_pred[[6]] <- refe_model(data_train, data_test)
p_refe_reg[6,] <- acc_out(refe_pred[[6]][[1]], Y_outcome)
p_refe_gbm[6,] <- acc_out(refe_pred[[6]][[2]], Y_outcome)
})

try({
gain_pred[[6]] <- gain_model(data_train, data_test)
p_gain_reg[6,] <- acc_out(gain_pred[[6]][[1]], Y_outcome)
p_gain_gbm[6,] <- acc_out(gain_pred[[6]][[2]], Y_outcome)
})

try({
cart_pred_i[[6]] <- cart_i_model(data_train, data_test)
p_cart_i[6,] <- acc_out(cart_pred_i[[6]], Y_outcome)
})

try({
# lazy_pred[[6]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[6,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[6,] <- acc_out(Y_outcome, Y_outcome)
})



#### subset 7

data_train <- tr_test_st(dat_list[[7]])[[1]]
data_test <- tr_test_st(dat_list[[7]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)


try({
brm_pred[[7]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[7,] <- acc_out(brm_pred[[7]][[1]], Y_outcome)
p_brm_gbm[7,] <- acc_out(brm_pred[[7]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[7]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[7,] <- acc_out(brm_n_ov_pred[[7]][[1]], Y_outcome)
p_brm_n_ov_gbm[7,] <- acc_out(brm_n_ov_pred[[7]][[2]], Y_outcome)
})

try({
listwise_pred[[7]] <- listwise_model(data_train, data_test)
p_listwise_reg[7,] <- acc_out(listwise_pred[[7]][[1]], Y_outcome)
p_listwise_gbm[7,] <- acc_out(listwise_pred[[7]][[2]], Y_outcome)
})

try({
columnwise_pred[[7]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[7,] <- acc_out(columnwise_pred[[7]][[1]], Y_outcome)
p_columnwise_gbm[7,] <- acc_out(columnwise_pred[[7]][[2]], Y_outcome)
})

try({
mi_pred[[7]] <- mi_model(data_train, data_test)
p_mi_reg[7,] <- acc_out(mi_pred[[7]][[1]], Y_outcome)
p_mi_gbm[7,] <- acc_out(mi_pred[[7]][[2]], Y_outcome)
})

try({
si_pred[[7]] <- si_model(data_train, data_test)
p_si_reg[7,] <- acc_out(si_pred[[7]][[1]], Y_outcome)
p_si_gbm[7,] <- acc_out(si_pred[[7]][[2]], Y_outcome)
})

try({
cart_pred[[7]] <- cart_model(data_train, data_test)
p_cart[7,] <- acc_out(cart_pred[[7]], Y_outcome)
})

try({
imsf_pred[[7]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[7,] <- acc_out(imsf_pred[[7]], Y_outcome)
})

try({
discom_pred[[7]] <- discom_model(num_blocks, data_train,data_test)
p_discom[7,] <- acc_out(discom_pred[[7]], Y_outcome)
})

try({
refe_pred[[7]] <- refe_model(data_train, data_test)
p_refe_reg[7,] <- acc_out(refe_pred[[7]][[1]], Y_outcome)
p_refe_gbm[7,] <- acc_out(refe_pred[[7]][[2]], Y_outcome)
})

try({
gain_pred[[7]] <- gain_model(data_train, data_test)
p_gain_reg[7,] <- acc_out(gain_pred[[7]][[1]], Y_outcome)
p_gain_gbm[7,] <- acc_out(gain_pred[[7]][[2]], Y_outcome)
})

try({
cart_pred_i[[7]] <- cart_i_model(data_train, data_test)
p_cart_i[7,] <- acc_out(cart_pred_i[[7]], Y_outcome)
})

try({
# lazy_pred[[7]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[7,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[7,] <- acc_out(Y_outcome, Y_outcome)
})


#### subset 8

data_train <- tr_test_st(dat_list[[8]])[[1]]
data_test <- tr_test_st(dat_list[[8]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)


try({
brm_pred[[8]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[8,] <- acc_out(brm_pred[[8]][[1]], Y_outcome)
p_brm_gbm[8,] <- acc_out(brm_pred[[8]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[8]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test,low_threshold = 0.05, n=10)
p_brm_n_ov_reg[8,] <- acc_out(brm_n_ov_pred[[8]][[1]], Y_outcome)
p_brm_n_ov_gbm[8,] <- acc_out(brm_n_ov_pred[[8]][[2]], Y_outcome)
})

try({
listwise_pred[[8]] <- listwise_model(data_train, data_test)
p_listwise_reg[8,] <- acc_out(listwise_pred[[8]][[1]], Y_outcome)
p_listwise_gbm[8,] <- acc_out(listwise_pred[[8]][[2]], Y_outcome)
})

try({
columnwise_pred[[8]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[8,] <- acc_out(columnwise_pred[[8]][[1]], Y_outcome)
p_columnwise_gbm[8,] <- acc_out(columnwise_pred[[8]][[2]], Y_outcome)
})

try({
mi_pred[[8]] <- mi_model(data_train, data_test)
p_mi_reg[8,] <- acc_out(mi_pred[[8]][[1]], Y_outcome)
p_mi_gbm[8,] <- acc_out(mi_pred[[8]][[2]], Y_outcome)
})

try({
si_pred[[8]] <- si_model(data_train, data_test)
p_si_reg[8,] <- acc_out(si_pred[[8]][[1]], Y_outcome)
p_si_gbm[8,] <- acc_out(si_pred[[8]][[2]], Y_outcome)
})

try({
cart_pred[[8]] <- cart_model(data_train, data_test)
p_cart[8,] <- acc_out(cart_pred[[8]], Y_outcome)
})

try({
imsf_pred[[8]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[8,] <- acc_out(imsf_pred[[8]], Y_outcome)
})

try({
discom_pred[[8]] <- discom_model(num_blocks, data_train,data_test)
p_discom[8,] <- acc_out(discom_pred[[8]], Y_outcome)
})

try({
refe_pred[[8]] <- refe_model(data_train, data_test)
p_refe_reg[8,] <- acc_out(refe_pred[[8]][[1]], Y_outcome)
p_refe_gbm[8,] <- acc_out(refe_pred[[8]][[2]], Y_outcome)
})

try({
gain_pred[[8]] <- gain_model(data_train, data_test)
p_gain_reg[8,] <- acc_out(gain_pred[[8]][[1]], Y_outcome)
p_gain_gbm[8,] <- acc_out(gain_pred[[8]][[2]], Y_outcome)
})

try({
cart_pred_i[[8]] <- cart_i_model(data_train, data_test)
p_cart_i[8,] <- acc_out(cart_pred_i[[8]], Y_outcome)
})

try({
# lazy_pred[[8]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[8,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[8,] <- acc_out(Y_outcome, Y_outcome)
})


#### subset 9

data_train <- tr_test_st(dat_list[[9]])[[1]]
data_test <- tr_test_st(dat_list[[9]])[[2]]
data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)
Y_outcome = data_test[,outcome]

missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )
num_blocks = num_of_blocks(missing_prop_val0.05)

try({
brm_pred[[9]] <- brm_predictions(num_blocks,data_train_x, data_train_y, data_test)
p_brm_reg[9,] <- acc_out(brm_pred[[9]][[1]], Y_outcome)
p_brm_gbm[9,] <- acc_out(brm_pred[[9]][[2]], Y_outcome)
})

try({
brm_n_ov_pred[[9]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[9,] <- acc_out(brm_n_ov_pred[[9]][[1]], Y_outcome)
p_brm_n_ov_gbm[9,] <- acc_out(brm_n_ov_pred[[9]][[2]], Y_outcome)
})

try({
listwise_pred[[9]] <- listwise_model(data_train, data_test)
p_listwise_reg[9,] <- acc_out(listwise_pred[[9]][[1]], Y_outcome)
p_listwise_gbm[9,] <- acc_out(listwise_pred[[9]][[2]], Y_outcome)
})

try({
columnwise_pred[[9]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[9,] <- acc_out(columnwise_pred[[9]][[1]], Y_outcome)
p_columnwise_gbm[9,] <- acc_out(columnwise_pred[[9]][[2]], Y_outcome)
})

try({
mi_pred[[9]] <- mi_model(data_train, data_test)
p_mi_reg[9,] <- acc_out(mi_pred[[9]][[1]], Y_outcome)
p_mi_gbm[9,] <- acc_out(mi_pred[[9]][[2]], Y_outcome)
})

try({
si_pred[[9]] <- si_model(data_train, data_test)
p_si_reg[9,] <- acc_out(si_pred[[9]][[1]], Y_outcome)
p_si_gbm[9,] <- acc_out(si_pred[[9]][[2]], Y_outcome)
})

try({
cart_pred[[9]] <- cart_model(data_train, data_test)
p_cart[9,] <- acc_out(cart_pred[[9]], Y_outcome)
})

try({
imsf_pred[[9]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[9,] <- acc_out(imsf_pred[[9]], Y_outcome)
})

try({
discom_pred[[9]] <- discom_model(num_blocks, data_train,data_test)
p_discom[9,] <- acc_out(discom_pred[[9]], Y_outcome)
})

try({
refe_pred[[9]] <- refe_model(data_train, data_test)
p_refe_reg[9,] <- acc_out(refe_pred[[9]][[1]], Y_outcome)
p_refe_gbm[9,] <- acc_out(refe_pred[[9]][[2]], Y_outcome)
})

try({
gain_pred[[9]] <- gain_model(data_train, data_test)
p_gain_reg[9,] <- acc_out(gain_pred[[9]][[1]], Y_outcome)
p_gain_gbm[9,] <- acc_out(gain_pred[[9]][[2]], Y_outcome)
})

try({
cart_pred_i[[9]] <- cart_i_model(data_train, data_test)
p_cart_i[9,] <- acc_out(cart_pred_i[[9]], Y_outcome)
})

try({
# lazy_pred[[9]] <- lazy_tree_model(data_train_x, data_train_y, data_test)
p_lazy_reg[9,] <- acc_out(Y_outcome, Y_outcome)
p_lazy_tree[9,] <- acc_out(Y_outcome, Y_outcome)
})


p_brm_reg
p_brm_gbm
p_brm_n_ov_reg
p_brm_n_ov_gbm
p_listwise_reg
p_listwise_gbm
p_columnwise_reg
p_columnwise_gbm
p_si_reg
p_si_gbm
p_mi_reg
p_mi_gbm
p_cart
p_imsf
p_discom
p_refe_reg
p_refe_gbm
p_gain_reg
p_gain_gbm

p_cart_i
p_lazy_reg
p_lazy_tree


acc_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                     brm= p_brm_reg[,1],
                     brm_nov = p_brm_n_ov_reg[,1],
                     imsf = p_imsf[,1],
                     listwise = p_listwise_reg[,1],
                     discom = p_discom[,1],
                     columnwise = p_columnwise_reg[,1],
                     refe = p_refe_reg[,1],
                     si = p_si_reg[,1],
                     gain = p_gain_reg[,1],
                     mi = p_mi_reg[,1],
                     lazy_reg = p_lazy_reg[,1] )

acc_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                       brm = p_brm_gbm[,1],
                       brm_nov = p_brm_n_ov_gbm[,1],
                       listwise = p_listwise_gbm[,1],
                       columnwise = p_columnwise_gbm[,1],
                       si = p_si_gbm[,1],
                       mi = p_mi_gbm[,1],
                       cart = p_cart[,1],
                       refe =  p_refe_gbm[,1],
                       gain =  p_gain_gbm[,1],
                       cart_i = p_cart_i[,1],
                       lazy_tree = p_lazy_tree[,1]) 



auc_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                     brm= p_brm_reg[,2],
                     brm_nov = p_brm_n_ov_reg[,2],
                     imsf = p_imsf[,2],
                     listwise = p_listwise_reg[,2],
                     discom = p_discom[,2],
                     columnwise = p_columnwise_reg[,2],
                     refe = p_refe_reg[,2],
                     si = p_si_reg[,2],
                     gain = p_gain_reg[,2],
                     mi = p_mi_reg[,2] ,
                     lazy_reg = p_lazy_reg[,2])

auc_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                       brm = p_brm_gbm[,2],
                       brm_nov = p_brm_n_ov_gbm[,2],
                       listwise = p_listwise_gbm[,2],
                       columnwise = p_columnwise_gbm[,2],
                       si = p_si_gbm[,2],
                       mi = p_mi_gbm[,2],
                       cart = p_cart[,2],
                       refe =  p_refe_gbm[,2],
                       gain =  p_gain_gbm[,2],
                       cart_i = p_cart_i[,2],
                       lazy_tree = p_lazy_tree[,2])


recall_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                        brm= p_brm_reg[,3],
                        brm_nov = p_brm_n_ov_reg[,3],
                        imsf = p_imsf[,3],
                        listwise = p_listwise_reg[,3],
                        discom = p_discom[,3],
                        columnwise = p_columnwise_reg[,3],
                        refe = p_refe_reg[,3],
                        si = p_si_reg[,3],
                        gain = p_gain_reg[,3],
                        mi = p_mi_reg[,3],
                        lazy_reg = p_lazy_reg[,3])

recall_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                          brm = p_brm_gbm[,3],
                          brm_nov = p_brm_n_ov_gbm[,3],
                          listwise = p_listwise_gbm[,3],
                          columnwise = p_columnwise_gbm[,3],
                          si = p_si_gbm[,3],
                          mi = p_mi_gbm[,3],
                          cart = p_cart[,3],
                          refe =  p_refe_gbm[,3],
                          gain =  p_gain_gbm[,3],
                          cart_i = p_cart_i[,3],
                          lazy_tree = p_lazy_tree[,3])



precision_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                           brm= p_brm_reg[,4],
                           brm_nov = p_brm_n_ov_reg[,4],
                           imsf = p_imsf[,4],
                           listwise = p_listwise_reg[,4],
                           discom = p_discom[,4],
                           columnwise = p_columnwise_reg[,4],
                           refe = p_refe_reg[,4],
                           si = p_si_reg[,4],
                           gain = p_gain_reg[,4],
                           mi = p_mi_reg[,4] ,
                           lazy_reg = p_lazy_reg[,4])

precision_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                             brm = p_brm_gbm[,4],
                             brm_nov = p_brm_n_ov_gbm[,4],
                             listwise = p_listwise_gbm[,4],
                             columnwise = p_columnwise_gbm[,4],
                             si = p_si_gbm[,4],
                             mi = p_mi_gbm[,4],
                             cart = p_cart[,4],
                             refe =  p_refe_gbm[,4],
                             gain =  p_gain_gbm[,4],
                             cart_i = p_cart_i[,4],
                             lazy_tree = p_lazy_tree[,4])


f1_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                    brm= p_brm_reg[,5],
                    brm_nov = p_brm_n_ov_reg[,5],
                    imsf = p_imsf[,5],
                    listwise = p_listwise_reg[,5],
                    discom = p_discom[,5],
                    columnwise = p_columnwise_reg[,5],
                    refe = p_refe_reg[,5],
                    si = p_si_reg[,5],
                    gain = p_gain_reg[,5],
                    mi = p_mi_reg[,5] ,
                    lazy_reg = p_lazy_reg[,5])

f1_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                      brm = p_brm_gbm[,5],
                      brm_nov = p_brm_n_ov_gbm[,5],
                      listwise = p_listwise_gbm[,5],
                      columnwise = p_columnwise_gbm[,5],
                      si = p_si_gbm[,5],
                      mi = p_mi_gbm[,5],
                      cart = p_cart[,5],
                      refe =  p_refe_gbm[,5],
                      gain =  p_gain_gbm[,5],
                      cart_i = p_cart_i[,5],
                      lazy_tree = p_lazy_tree[,5])

#write.csv(acc_reg, "/root/capsule/results/acc_reg_adult.csv",row.names=F)
#write.csv(acc_tree, "/root/capsule/results/acc_tree_adult.csv",row.names=F)
write.csv(auc_reg, "/root/capsule/results/auc_reg_adult.csv",row.names=F)
write.csv(auc_tree, "/root/capsule/results/auc_tree_adult.csv",row.names=F)
#write.csv(recall_reg, "/root/capsule/results/recall_reg_adult.csv",row.names=F)
#write.csv(recall_tree, "/root/capsule/results/recall_tree_adult.csv",row.names=F)
#write.csv(precision_reg, "/root/capsule/results/precision_reg_adult.csv",row.names=F)
#write.csv(precision_tree, "/root/capsule/results/precision_tree_adult.csv",row.names=F)
write.csv(f1_reg, "/root/capsule/results/f1_reg_adult.csv",row.names=F)
write.csv(f1_tree, "/root/capsule/results/f1_tree_adult.csv",row.names=F)