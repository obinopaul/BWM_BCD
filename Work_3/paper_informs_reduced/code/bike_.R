## Note: We have commented lazy decision trees benchmark in
### the reproducibility run as it takes more than 12 hours to ### run the module whereas codeocean has limite resources. 

gen_missing_bike <- function(bike,propMissing)
{  
  set.seed(1234)
  miss_1 <- sample(nrow(bike),propMissing*nrow(bike),replace=FALSE )
  set.seed(123)
  miss_2 <- sample(nrow(bike),propMissing*nrow(bike),replace=FALSE )
  
  bike_mis <- bike  
  
  ### Generate missing values
  
  bike_mis[miss_1,names(bike_mis) %in% c('windspeed','hum','ToD','weekday')] <- NA
  bike_mis[miss_2,names(bike_mis) %in% c('hr','temp','temp2','weathersit')] <- NA
  
  ## Add random noise (only little = 5% ) ## Preserve the noise dataset (important)
  miss_3 <- bike_mis[,names(bike_mis)%in% c('ToD','hr','weathersit','windspeed','temp','temp2','hum','weekday')]
  miss_4 <- bike_mis[,!names(bike_mis)%in% c('ToD','hr','weathersit','windspeed','temp','temp2','hum','weekday')]
    
  for(col in 1:ncol(miss_3))
  {
    m <- 12 + col  ##For reproducibility
    set.seed(m)
    miss_temp <- sample(nrow(miss_3),nrow(miss_3)*0.05,replace=FALSE )
    miss_3[miss_temp,col] <- NA 
  }
  
  bike_mis_n <- cbind(miss_3,miss_4)
  return(bike_mis_n)
}

reg_train <- function(data_in)
{
  X <- data_in[,names(data_in) %in% names(data_train_x) ]
  Y <- data_in[,names(data_in) %in% names(data_train_y) ]
  #datam <- data.table(X,Y)
  fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))
  # reg <- stepAIC(glm( fmla, data=data_in,family="poisson"),direction="both",trace=FALSE)
  try( reg <- glm.nb( fmla, data=data_in ) )
  
  if(all(is.na(reg)))
    {
    reg <-lm( fmla, data=data_in)
    }
  # reg <- lm( fmla, data=datam)
  return(reg)
}

tree_train <- function(data_in)
{
  X <- data_in[,names(data_in) %in% names(data_train_x) ]
  Y <- data_in[,names(data_in) %in% names(data_train_y) ]
  datam <- data.table(X,Y) 
  fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))
  #reg <- gbm( fmla, data=datam ,verbose=FALSE,n.trees = 500,shrinkage=0.1,interaction.depth = 3)
  reg <- gbm( fmla, data=datam ,distribution ="poisson" ,verbose=FALSE,n.trees = 500)
  return(reg)
}

## Get the data and do some pre-processing
data_all <- read.csv("/root/capsule/data/bike_all.csv")
bike <- data_all[,!names(data_all) %in% c("instant","dteday","atemp","yr","workingday","holiday")] 
bike$weekday <- as.factor(bike$weekday)
bike$weathersit[bike$weathersit==4] <- 3
bike$weathersit <- as.factor(bike$weathersit)
bike$season <- as.factor(bike$season)
bike$ToD <- cut(bike$hr, breaks = c(-0.01,5,10,16,19,24))
## Renormalize
bike$temp <- bike$temp*(39+8) -8 
bike$temp2 <- bike$temp^2

dat_list <- list()
dat_list[[1]] <- gen_missing_bike(bike,0.10) 
dat_list[[2]] <- gen_missing_bike(bike,0.20)
dat_list[[3]] <- gen_missing_bike(bike,0.30) 
dat_list[[4]] <- gen_missing_bike(bike,0.40)
dat_list[[5]] <- gen_missing_bike(bike,0.50) 
dat_list[[6]] <- gen_missing_bike(bike,0.60)
dat_list[[7]] <- gen_missing_bike(bike,0.70)
dat_list[[8]] <- gen_missing_bike(bike,0.80) 
dat_list[[9]] <- gen_missing_bike(bike,0.90) 
### 50% missing:
data_train <- tr_test_st(dat_list[[5]])[[1]]
data_test <- tr_test_st(dat_list[[5]])[[2]]

####
## Change this for each dataset
####
outcome = "cnt"
###
###

data_train_y <- data.frame(data_train[,outcome])
data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
names(data_train_y) = outcome 
data_test_imputed <- impute_test(data_train,data_test)
data_train_imputed <- impute_manual(data_train)

data_train_no_miss <- tr_test_st(bike)[[1]]
data_test_no_miss <- tr_test_st(bike)[[2]]

# system.time( missing_prop_val0.01 <- brm_num_blocks(data_train_x,low_threshold = 0.01 )  ) 
# #plot(missing_prop_val0.01)
# num_of_blocks(missing_prop_val0.01)
system.time( missing_prop_val0.05 <- brm_num_blocks(data_train_x,low_threshold = 0.05 )  ) 
#plot(missing_prop_val0.05,type='l')
#num_of_blocks(missing_prop_val0.05)
# system.time( missing_prop_val0.10 <- brm_num_blocks(data_train_x,low_threshold = 0.10 )  ) 
# #plot(missing_prop_val0.10)
# num_of_blocks(missing_prop_val0.10)

num_blocks = num_of_blocks(missing_prop_val0.05)


p_brm_reg <- matrix(NA, nrow = 9, ncol = 4)
p_brm_gbm <- matrix(NA, nrow = 9, ncol = 4)
p_brm_n_ov_reg <- matrix(NA, nrow = 9, ncol = 4)
p_brm_n_ov_gbm <- matrix(NA, nrow = 9, ncol = 4)
p_listwise_reg <- matrix(NA, nrow = 9, ncol = 4)
p_listwise_gbm <- matrix(NA, nrow = 9, ncol = 4)
p_columnwise_reg <- matrix(NA, nrow = 9, ncol = 4)
p_columnwise_gbm <- matrix(NA, nrow = 9, ncol = 4)
p_si_reg <- matrix(NA, nrow = 9, ncol = 4)
p_si_gbm <- matrix(NA, nrow = 9, ncol = 4)
p_mi_reg <- matrix(NA, nrow = 9, ncol = 4)
p_mi_gbm <- matrix(NA, nrow = 9, ncol = 4)
p_cart <- matrix(NA, nrow = 9, ncol = 4)

p_imsf <- matrix(NA, nrow = 9, ncol = 4)
p_discom <- matrix(NA, nrow = 9, ncol = 4)
p_refe_reg <- matrix(NA, nrow = 9, ncol = 4)
p_refe_gbm <- matrix(NA, nrow = 9, ncol = 4)
p_gain_reg <- matrix(NA, nrow = 9, ncol = 4)
p_gain_gbm <- matrix(NA, nrow = 9, ncol = 4)

p_cart_i <- matrix(NA, nrow = 9, ncol = 4)
p_lazy_reg <- matrix(NA, nrow = 9, ncol = 4)
p_lazy_tree <- matrix(NA, nrow = 9, ncol = 4)


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
p_brm_reg[1,] <- predperf(brm_pred[[1]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[1,] <- predperf(brm_pred[[1]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[1]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[1,] <- predperf(brm_n_ov_pred[[1]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[1,] <- predperf(brm_n_ov_pred[[1]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[1]] <- listwise_model(data_train, data_test)
p_listwise_reg[1,] <- predperf(listwise_pred[[1]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[1,] <- predperf(listwise_pred[[1]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[1]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[1,] <- predperf(columnwise_pred[[1]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[1,] <- predperf(columnwise_pred[[1]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[1]] <- mi_model(data_train, data_test)
p_mi_reg[1,] <- predperf(mi_pred[[1]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[1,] <- predperf(mi_pred[[1]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[1]] <- si_model(data_train, data_test)
p_si_reg[1,] <- predperf(si_pred[[1]][[1]], Y_outcome,count_ind=1)
p_si_gbm[1,] <- predperf(si_pred[[1]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[1]] <- cart_model(data_train, data_test, type = "anova")
p_cart[1,] <- predperf(cart_pred[[1]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[1]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[1,] <- predperf(imsf_pred[[1]], Y_outcome,count_ind=1)
})

try({
discom_pred[[1]] <- discom_model(num_blocks, data_train,data_test)
p_discom[1,] <- predperf(discom_pred[[1]], Y_outcome,count_ind=1)
})

try({
refe_pred[[1]] <- refe_model(data_train, data_test)
p_refe_reg[1,] <- predperf(refe_pred[[1]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[1,] <- predperf(refe_pred[[1]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[1]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[1,] <- predperf(cart_pred_i[[1]], Y_outcome,count_ind=1)
})

try({
gain_pred[[1]] <- gain_model(data_train, data_test)
p_gain_reg[1,] <- predperf(gain_pred[[1]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[1,] <- predperf(gain_pred[[1]][[2]], Y_outcome,count_ind=1)
})


#system.time(lazy_pred[[1]] <- lazy_tree_model(data_train_x, #data_train_y, data_test, type = "anova") )
## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
#p_lazy_reg[1,] <- predperf(lazy_pred[[1]][[1]], Y_outcome,count_ind=1)
#p_lazy_tree[1,] <- predperf(lazy_pred[[1]][[2]], Y_outcome,count_ind=1)
p_lazy_reg[1,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[1,] <- predperf(Y_outcome, Y_outcome,count_ind=1)


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
p_brm_reg[2,] <- predperf(brm_pred[[2]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[2,] <- predperf(brm_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[2]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[2,] <- predperf(brm_n_ov_pred[[2]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[2,] <- predperf(brm_n_ov_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[2]] <- listwise_model(data_train, data_test)
p_listwise_reg[2,] <- predperf(listwise_pred[[2]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[2,] <- predperf(listwise_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[2]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[2,] <- predperf(columnwise_pred[[2]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[2,] <- predperf(columnwise_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[2]] <- mi_model(data_train, data_test)
p_mi_reg[2,] <- predperf(mi_pred[[2]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[2,] <- predperf(mi_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[2]] <- si_model(data_train, data_test)
p_si_reg[2,] <- predperf(si_pred[[2]][[1]], Y_outcome,count_ind=1)
p_si_gbm[2,] <- predperf(si_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[2]] <- cart_model(data_train, data_test, type = "anova")
p_cart[2,] <- predperf(cart_pred[[2]], Y_outcome,count_ind=1)
})


try({
imsf_pred[[2]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[2,] <- predperf(imsf_pred[[2]], Y_outcome,count_ind=1)
})

try({
discom_pred[[2]] <- discom_model(num_blocks, data_train,data_test)
p_discom[2,] <- predperf(discom_pred[[2]], Y_outcome,count_ind=1)
})

try({
refe_pred[[2]] <- refe_model(data_train, data_test)
p_refe_reg[2,] <- predperf(refe_pred[[2]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[2,] <- predperf(refe_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[2]] <- gain_model(data_train, data_test)
p_gain_reg[2,] <- predperf(gain_pred[[2]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[2,] <- predperf(gain_pred[[2]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[2]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[2,] <- predperf(cart_pred_i[[2]], Y_outcome,count_ind=1)
})


#system.time(lazy_pred[[2]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )

## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
#p_lazy_reg[2,] <- predperf(lazy_pred[[2]][[1]], Y_outcome,count_ind=1)
#p_lazy_tree[2,] <- predperf(lazy_pred[[2]][[2]], Y_outcome,count_ind=1)
p_lazy_reg[2,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[2,] <- predperf(Y_outcome, Y_outcome,count_ind=1)

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
p_brm_reg[3,] <- predperf(brm_pred[[3]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[3,] <- predperf(brm_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[3]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[3,] <- predperf(brm_n_ov_pred[[3]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[3,] <- predperf(brm_n_ov_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[3]] <- listwise_model(data_train, data_test)
p_listwise_reg[3,] <- predperf(listwise_pred[[3]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[3,] <- predperf(listwise_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[3]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[3,] <- predperf(columnwise_pred[[3]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[3,] <- predperf(columnwise_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[3]] <- mi_model(data_train, data_test)
p_mi_reg[3,] <- predperf(mi_pred[[3]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[3,] <- predperf(mi_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[3]] <- si_model(data_train, data_test)
p_si_reg[3,] <- predperf(si_pred[[3]][[1]], Y_outcome,count_ind=1)
p_si_gbm[3,] <- predperf(si_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[3]] <- cart_model(data_train, data_test, type = "anova")
p_cart[3,] <- predperf(cart_pred[[3]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[3]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[3,] <- predperf(imsf_pred[[3]], Y_outcome,count_ind=1)
})

try({
discom_pred[[3]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[3,] <- predperf(discom_pred[[3]], Y_outcome,count_ind=1) 
})

try({
refe_pred[[3]] <- refe_model(data_train, data_test)
p_refe_reg[3,] <- predperf(refe_pred[[3]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[3,] <- predperf(refe_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[3]] <- gain_model(data_train, data_test)
p_gain_reg[3,] <- predperf(gain_pred[[3]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[3,] <- predperf(gain_pred[[3]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[3]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[3,] <- predperf(cart_pred_i[[3]], Y_outcome,count_ind=1)
})


#system.time(lazy_pred[[3]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )

## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
p_lazy_reg[3,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[3,] <- predperf(Y_outcome, Y_outcome,count_ind=1)


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
p_brm_reg[4,] <- predperf(brm_pred[[4]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[4,] <- predperf(brm_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[4]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[4,] <- predperf(brm_n_ov_pred[[4]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[4,] <- predperf(brm_n_ov_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[4]] <- listwise_model(data_train, data_test)
p_listwise_reg[4,] <- predperf(listwise_pred[[4]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[4,] <- predperf(listwise_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[4]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[4,] <- predperf(columnwise_pred[[4]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[4,] <- predperf(columnwise_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[4]] <- mi_model(data_train, data_test)
p_mi_reg[4,] <- predperf(mi_pred[[4]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[4,] <- predperf(mi_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[4]] <- si_model(data_train, data_test)
p_si_reg[4,] <- predperf(si_pred[[4]][[1]], Y_outcome,count_ind=1)
p_si_gbm[4,] <- predperf(si_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[4]] <- cart_model(data_train, data_test, type = "anova")
p_cart[4,] <- predperf(cart_pred[[4]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[4]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[4,] <- predperf(imsf_pred[[4]], Y_outcome,count_ind=1)
})

try({
discom_pred[[4]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[4,] <- predperf(discom_pred[[4]], Y_outcome,count_ind=1) 
})

try({
refe_pred[[4]] <- refe_model(data_train, data_test)
p_refe_reg[4,] <- predperf(refe_pred[[4]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[4,] <- predperf(refe_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[4]] <- gain_model(data_train, data_test)
p_gain_reg[4,] <- predperf(gain_pred[[4]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[4,] <- predperf(gain_pred[[4]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[4]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[4,] <- predperf(cart_pred_i[[4]], Y_outcome,count_ind=1)
})


#system.time(lazy_pred[[4]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )

## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
p_lazy_reg[4,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[4,] <- predperf(Y_outcome, Y_outcome,count_ind=1)

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
p_brm_reg[5,] <- predperf(brm_pred[[5]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[5,] <- predperf(brm_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[5]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[5,] <- predperf(brm_n_ov_pred[[5]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[5,] <- predperf(brm_n_ov_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[5]] <- listwise_model(data_train, data_test)
p_listwise_reg[5,] <- predperf(listwise_pred[[5]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[5,] <- predperf(listwise_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[5]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[5,] <- predperf(columnwise_pred[[5]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[5,] <- predperf(columnwise_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[5]] <- mi_model(data_train, data_test)
p_mi_reg[5,] <- predperf(mi_pred[[5]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[5,] <- predperf(mi_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[5]] <- si_model(data_train, data_test)
p_si_reg[5,] <- predperf(si_pred[[5]][[1]], Y_outcome,count_ind=1)
p_si_gbm[5,] <- predperf(si_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[5]] <- cart_model(data_train, data_test, type = "anova")
p_cart[5,] <- predperf(cart_pred[[5]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[5]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[5,] <- predperf(imsf_pred[[5]], Y_outcome,count_ind=1)
})

try({
discom_pred[[5]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[5,] <- predperf(discom_pred[[5]], Y_outcome,count_ind=1) 
})

try({
refe_pred[[5]] <- refe_model(data_train, data_test)
p_refe_reg[5,] <- predperf(refe_pred[[5]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[5,] <- predperf(refe_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[5]] <- gain_model(data_train, data_test)
p_gain_reg[5,] <- predperf(gain_pred[[5]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[5,] <- predperf(gain_pred[[5]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[5]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[5,] <- predperf(cart_pred_i[[5]], Y_outcome,count_ind=1)
})

#system.time(lazy_pred[[5]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )
## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
p_lazy_reg[5,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[5,] <- predperf(Y_outcome, Y_outcome,count_ind=1)


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
p_brm_reg[6,] <- predperf(brm_pred[[6]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[6,] <- predperf(brm_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[6]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[6,] <- predperf(brm_n_ov_pred[[6]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[6,] <- predperf(brm_n_ov_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[6]] <- listwise_model(data_train, data_test)
p_listwise_reg[6,] <- predperf(listwise_pred[[6]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[6,] <- predperf(listwise_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[6]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[6,] <- predperf(columnwise_pred[[6]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[6,] <- predperf(columnwise_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[6]] <- mi_model(data_train, data_test)
p_mi_reg[6,] <- predperf(mi_pred[[6]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[6,] <- predperf(mi_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[6]] <- si_model(data_train, data_test)
p_si_reg[6,] <- predperf(si_pred[[6]][[1]], Y_outcome,count_ind=1)
p_si_gbm[6,] <- predperf(si_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[6]] <- cart_model(data_train, data_test, type = "anova")
p_cart[6,] <- predperf(cart_pred[[6]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[6]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[6,] <- predperf(imsf_pred[[6]], Y_outcome,count_ind=1)
})

try({
discom_pred[[6]] <- discom_model(num_blocks, data_train,data_test) 
p_discom[6,] <- predperf(discom_pred[[6]], Y_outcome,count_ind=1) 
})

try({
refe_pred[[6]] <- refe_model(data_train, data_test)
p_refe_reg[6,] <- predperf(refe_pred[[6]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[6,] <- predperf(refe_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[6]] <- gain_model(data_train, data_test)
p_gain_reg[6,] <- predperf(gain_pred[[6]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[6,] <- predperf(gain_pred[[6]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[6]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[6,] <- predperf(cart_pred_i[[6]], Y_outcome,count_ind=1)
})

#system.time(lazy_pred[[6]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )
## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:

p_lazy_reg[6,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[6,] <- predperf(Y_outcome, Y_outcome,count_ind=1)

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
p_brm_reg[7,] <- predperf(brm_pred[[7]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[7,] <- predperf(brm_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[7]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[7,] <- predperf(brm_n_ov_pred[[7]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[7,] <- predperf(brm_n_ov_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[7]] <- listwise_model(data_train, data_test)
p_listwise_reg[7,] <- predperf(listwise_pred[[7]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[7,] <- predperf(listwise_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[7]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[7,] <- predperf(columnwise_pred[[7]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[7,] <- predperf(columnwise_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[7]] <- mi_model(data_train, data_test)
p_mi_reg[7,] <- predperf(mi_pred[[7]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[7,] <- predperf(mi_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[7]] <- si_model(data_train, data_test)
p_si_reg[7,] <- predperf(si_pred[[7]][[1]], Y_outcome,count_ind=1)
p_si_gbm[7,] <- predperf(si_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[7]] <- cart_model(data_train, data_test, type = "anova")
p_cart[7,] <- predperf(cart_pred[[7]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[7]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[7,] <- predperf(imsf_pred[[7]], Y_outcome,count_ind=1)
})

try({
discom_pred[[7]] <- discom_model(num_blocks, data_train,data_test)
p_discom[7,] <- predperf(discom_pred[[7]], Y_outcome,count_ind=1)
})

try({
refe_pred[[7]] <- refe_model(data_train, data_test)
p_refe_reg[7,] <- predperf(refe_pred[[7]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[7,] <- predperf(refe_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[7]] <- gain_model(data_train, data_test)
p_gain_reg[7,] <- predperf(gain_pred[[7]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[7,] <- predperf(gain_pred[[7]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[7]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[7,] <- predperf(cart_pred_i[[7]], Y_outcome,count_ind=1)
})

#system.time(lazy_pred[[7]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )
## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
p_lazy_reg[7,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[7,] <- predperf(Y_outcome, Y_outcome,count_ind=1)


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
p_brm_reg[8,] <- predperf(brm_pred[[8]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[8,] <- predperf(brm_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[8]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test,low_threshold = 0.05, n=10)
p_brm_n_ov_reg[8,] <- predperf(brm_n_ov_pred[[8]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[8,] <- predperf(brm_n_ov_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[8]] <- listwise_model(data_train, data_test)
p_listwise_reg[8,] <- predperf(listwise_pred[[8]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[8,] <- predperf(listwise_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[8]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[8,] <- predperf(columnwise_pred[[8]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[8,] <- predperf(columnwise_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[8]] <- mi_model(data_train, data_test)
p_mi_reg[8,] <- predperf(mi_pred[[8]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[8,] <- predperf(mi_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[8]] <- si_model(data_train, data_test)
p_si_reg[8,] <- predperf(si_pred[[8]][[1]], Y_outcome,count_ind=1)
p_si_gbm[8,] <- predperf(si_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[8]] <- cart_model(data_train, data_test, type = "anova")
p_cart[8,] <- predperf(cart_pred[[8]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[8]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[8,] <- predperf(imsf_pred[[8]], Y_outcome,count_ind=1)
})

try({
discom_pred[[8]] <- discom_model(num_blocks, data_train,data_test)
p_discom[8,] <- predperf(discom_pred[[8]], Y_outcome,count_ind=1)
})

try({
refe_pred[[8]] <- refe_model(data_train, data_test)
p_refe_reg[8,] <- predperf(refe_pred[[8]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[8,] <- predperf(refe_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[8]] <- gain_model(data_train, data_test)
p_gain_reg[8,] <- predperf(gain_pred[[8]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[8,] <- predperf(gain_pred[[8]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[8]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[8,] <- predperf(cart_pred_i[[8]], Y_outcome,count_ind=1)
})


#system.time(lazy_pred[[8]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )
## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
p_lazy_reg[8,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[8,] <- predperf(Y_outcome, Y_outcome,count_ind=1)

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
p_brm_reg[9,] <- predperf(brm_pred[[9]][[1]], Y_outcome,count_ind=1)
p_brm_gbm[9,] <- predperf(brm_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
brm_n_ov_pred[[9]] <- brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test)
p_brm_n_ov_reg[9,] <- predperf(brm_n_ov_pred[[9]][[1]], Y_outcome,count_ind=1)
p_brm_n_ov_gbm[9,] <- predperf(brm_n_ov_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
listwise_pred[[9]] <- listwise_model(data_train, data_test)
p_listwise_reg[9,] <- predperf(listwise_pred[[9]][[1]], Y_outcome,count_ind=1)
p_listwise_gbm[9,] <- predperf(listwise_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
columnwise_pred[[9]] <- columnwise_model(data_train, data_test)
p_columnwise_reg[9,] <- predperf(columnwise_pred[[9]][[1]], Y_outcome,count_ind=1)
p_columnwise_gbm[9,] <- predperf(columnwise_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
mi_pred[[9]] <- mi_model(data_train, data_test)
p_mi_reg[9,] <- predperf(mi_pred[[9]][[1]], Y_outcome,count_ind=1)
p_mi_gbm[9,] <- predperf(mi_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
si_pred[[9]] <- si_model(data_train, data_test)
p_si_reg[9,] <- predperf(si_pred[[9]][[1]], Y_outcome,count_ind=1)
p_si_gbm[9,] <- predperf(si_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred[[9]] <- cart_model(data_train, data_test, type = "anova")
p_cart[9,] <- predperf(cart_pred[[9]], Y_outcome,count_ind=1)
})

try({
imsf_pred[[9]] <- imsf_model(num_blocks, data_train,data_test)
p_imsf[9,] <- predperf(imsf_pred[[9]], Y_outcome,count_ind=1)
})

try({
discom_pred[[9]] <- discom_model(num_blocks, data_train,data_test)
p_discom[9,] <- predperf(discom_pred[[9]], Y_outcome,count_ind=1)
})

try({
refe_pred[[9]] <- refe_model(data_train, data_test)
p_refe_reg[9,] <- predperf(refe_pred[[9]][[1]], Y_outcome,count_ind=1)
p_refe_gbm[9,] <- predperf(refe_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
gain_pred[[9]] <- gain_model(data_train, data_test)
p_gain_reg[9,] <- predperf(gain_pred[[9]][[1]], Y_outcome,count_ind=1)
p_gain_gbm[9,] <- predperf(gain_pred[[9]][[2]], Y_outcome,count_ind=1)
})

try({
cart_pred_i[[9]] <- cart_i_model(data_train, data_test, type = "anova")
p_cart_i[9,] <- predperf(cart_pred_i[[9]], Y_outcome,count_ind=1)
})


#system.time(lazy_pred[[9]] <- lazy_tree_model(data_train_x, data_train_y, data_test, type = "anova") )
## Due to time constraint, we do not run lazy decision tree
#### Placeholder for this benchmark:
p_lazy_reg[9,] <- predperf(Y_outcome, Y_outcome,count_ind=1)
p_lazy_tree[9,] <- predperf(Y_outcome, Y_outcome,count_ind=1)


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


rmse_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                      brm= p_brm_reg[,1],
                     brm_nov = p_brm_n_ov_reg[,1],
                     imsf = p_imsf[,1],
                     listwise = p_listwise_reg[,1],
                     discom = p_discom[,1],
                     columnwise = p_columnwise_reg[,1],
                     refe = p_refe_reg[,1],
                     si = p_si_reg[,1],
                     gain = p_gain_reg[,1],
                     mi = p_mi_reg[,1] ,
                     lazy_reg = p_lazy_reg[,1])

rmse_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
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


mae_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
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

mae_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
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


smape_reg = data.frame(miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                     brm= p_brm_reg[,3],
                     brm_nov = p_brm_n_ov_reg[,3],
                     imsf = p_imsf[,3],
                     listwise = p_listwise_reg[,3],
                     discom = p_discom[,3],
                     columnwise = p_columnwise_reg[,3],
                     refe = p_refe_reg[,3],
                     si = p_si_reg[,3],
                     gain = p_gain_reg[,3],
                     mi = p_mi_reg[,3] ,
                     lazy_reg = p_lazy_reg[,3])

smape_tree = data.frame (miss_prop = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
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


write.csv(rmse_reg, "/root/capsule/results/rmse_reg_bike.csv",row.names=F)
write.csv(rmse_tree, "/root/capsule/results/rmse_tree_bike.csv",row.names=F)
#write.csv(mae_reg, "/root/capsule/results/mae_reg_bike.csv",row.names=F)
#write.csv(mae_tree, "/root/capsule/results/mae_tree_bike.csv",row.names=F)
write.csv(smape_reg, "/root/capsule/results/smape_reg_bike.csv",row.names=F)
write.csv(smape_tree, "/root/capsule/results/smape_tree_bike.csv",row.names=F)
