### Parts of the MI and lazy tree are set to NA to avoid the code running for more than 24 hours. 
## On uncommenting them, the results in the paper can be replicated fully. 

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
  datam <- data.table(X,Y)
  fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))
  # reg <- lm( fmla, data=datam)
  reg <- glm.nb(fmla, data=data_in)
  return(reg)
}

## We test scalability only for linear models, hence running a dummy function for nonlinear modeling:
tree_train <- function(data_in)
{
  return(1)
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
dat_list[[1]] <- gen_missing_bike(bike,0) 
dat_list[[2]] <- gen_missing_bike(bike,0.05)
dat_list[[3]] <- gen_missing_bike(bike,0.10) 
dat_list[[4]] <- gen_missing_bike(bike,0.15)
dat_list[[5]] <- gen_missing_bike(bike,0.20) 
dat_list[[6]] <- gen_missing_bike(bike,0.25)
dat_list[[7]] <- gen_missing_bike(bike,0.5)
dat_list[[8]] <- gen_missing_bike(bike,0.75) 

### 50% missing:
data_train <- tr_test_st(dat_list[[7]])[[1]]
data_test <- tr_test_st(dat_list[[7]])[[2]]

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

num_blocks = 4  ## We assume this as it is a one time computation for a given dataset and does not factor into model training process

bike_mis_n <- dat_list[[7]]

perturb <- function(data_in)
{
  data_in_ind <- sapply(data_in, is.numeric)
  data_in[,data_in_ind] <- lapply(data_in[,data_in_ind], jitter)
  data_in$cnt = round(data_in$cnt)
  # data_in[,data_in_ind] <- addNoise(data_in[,data_in_ind], variables = NULL, noise = 150, method = "additive")
  return(data_in)
}

### Here, we take different data sizes into consideration ##
sc_data_list <- list()
set.seed(12345)
sc_data_list[[1]] <- bike_mis_n[sample(nrow(bike_mis_n),5000,replace=TRUE ),]
sc_data_list[[1]] = perturb(sc_data_list[[1]] )
set.seed(12345)
sc_data_list[[2]] <- bike_mis_n[sample(nrow(bike_mis_n),10000,replace=TRUE ),]
sc_data_list[[2]] = perturb(sc_data_list[[2]] )
set.seed(12345)
sc_data_list[[3]] <- bike_mis_n[sample(nrow(bike_mis_n),50000,replace=TRUE ),]
sc_data_list[[3]] = perturb(sc_data_list[[3]] )
set.seed(12345)
sc_data_list[[4]] <- bike_mis_n[sample(nrow(bike_mis_n),100000,replace=TRUE ),]
sc_data_list[[4]] = perturb(sc_data_list[[4]] )
set.seed(12345)
sc_data_list[[5]] <- bike_mis_n[sample(nrow(bike_mis_n),500000,replace=TRUE ),]
sc_data_list[[5]] = perturb(sc_data_list[[5]] )
set.seed(12345)
sc_data_list[[6]] <- bike_mis_n[sample(nrow(bike_mis_n),1000000,replace=TRUE ),]
sc_data_list[[6]] = perturb(sc_data_list[[6]] )
# set.seed(12345)
# sc_data_list[[7]] <- bike_mis_n[sample(nrow(bike_mis_n),5000000,replace=TRUE ),]
# sc_data_list[[7]] = perturb(sc_data_list[[7]] )

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
lazy_pred <- list()

brm_time <- list()
brm_n_ov_time <- list()
listwise_time <- list()
columnwise_time <- list()
mi_time <- list()
si_time <- list()
cart_time <- list()
imsf_time <- list()
discom_time <- list()
refe_time <- list()
gain_time <- list()
lazy_time <- list()

data_prep_brm <- function(data_in)
{ 
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  data_train_y <- data.frame(data_train[,outcome])
  data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
  names(data_train_y) = outcome
  return(brm_predictions(num_blocks,data_train_x, data_train_y, data_test, low_threshold = 0.05, n=1))
}

brm_time <- c(
  system.time(brm_pred[[1]] <- data_prep_brm(sc_data_list[[1]])[[1]]  )[3],
  system.time(brm_pred[[2]]  <- data_prep_brm(sc_data_list[[2]])[[1]] )[3],
  system.time(brm_pred[[3]]  <- data_prep_brm(sc_data_list[[3]])[[1]] )[3],
  system.time(brm_pred[[4]]  <- data_prep_brm(sc_data_list[[4]])[[1]] )[3],
  system.time(brm_pred[[5]]  <- data_prep_brm(sc_data_list[[5]])[[1]] )[3],
  system.time(brm_pred[[6]]  <- data_prep_brm(sc_data_list[[6]])[[1]] )[3]
)

data_prep_brm_n_ov <- function(data_in)
{ 
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  data_train_y <- data.frame(data_train[,outcome])
  data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
  names(data_train_y) = outcome
  return(brm_predictions_n_ov(num_blocks,data_train_x, data_train_y, data_test, low_threshold = 0.05, n=1))
}

brm_n_ov_time <- c(
  system.time(brm_n_ov_pred[[1]] <- data_prep_brm_n_ov(sc_data_list[[1]])[[1]]  )[3],
  system.time(brm_n_ov_pred[[2]]  <- data_prep_brm_n_ov(sc_data_list[[2]])[[1]] )[3],
  system.time(brm_n_ov_pred[[3]]  <- data_prep_brm_n_ov(sc_data_list[[3]])[[1]] )[3],
  system.time(brm_n_ov_pred[[4]]  <- data_prep_brm_n_ov(sc_data_list[[4]])[[1]] )[3],
  system.time(brm_n_ov_pred[[5]]  <- data_prep_brm_n_ov(sc_data_list[[5]])[[1]] )[3],
  system.time(brm_n_ov_pred[[6]]  <- data_prep_brm_n_ov(sc_data_list[[6]])[[1]] )[3]
)

## Listwise
data_prep_listwise <- function(data_in)
{
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  return(listwise_model(data_train, data_test))
}

listwise_time <- c(
  system.time(temp <- data_prep_listwise(sc_data_list[[1]])[[1]]  )[3],
  system.time(temp  <- data_prep_listwise(sc_data_list[[2]])[[1]] )[3],
  system.time(temp  <- data_prep_listwise(sc_data_list[[3]])[[1]] )[3],
  system.time(temp  <- data_prep_listwise(sc_data_list[[4]])[[1]] )[3],
  system.time(temp  <- data_prep_listwise(sc_data_list[[5]])[[1]] )[3],
  system.time(temp  <- data_prep_listwise(sc_data_list[[6]])[[1]] )[3]
)

## Columnwise
data_prep_columnwise <- function(data_in)
{
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  return(columnwise_model(data_train, data_test))
}

columnwise_time <- c(
  system.time(temp <- data_prep_columnwise(sc_data_list[[1]])[[1]]  )[3],
  system.time(temp  <- data_prep_columnwise(sc_data_list[[2]])[[1]] )[3],
  system.time(temp  <- data_prep_columnwise(sc_data_list[[3]])[[1]] )[3],
  system.time(temp  <- data_prep_columnwise(sc_data_list[[4]])[[1]] )[3],
  system.time(temp  <- data_prep_columnwise(sc_data_list[[5]])[[1]] )[3],
  system.time(temp  <- data_prep_columnwise(sc_data_list[[6]])[[1]] )[3]
)

## GAIN
data_prep_gain <- function(data_in)
{
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  return(gain_model(data_train, data_test))
}

gain_time <- c(
  system.time(temp <- data_prep_gain(sc_data_list[[1]])[[1]]  )[3],
  system.time(temp  <- data_prep_gain(sc_data_list[[2]])[[1]] )[3],
  system.time(temp  <- data_prep_gain(sc_data_list[[3]])[[1]] )[3],
  system.time(temp  <- data_prep_gain(sc_data_list[[4]])[[1]] )[3],
  system.time(temp  <- data_prep_gain(sc_data_list[[5]])[[1]] )[3],
  system.time(temp  <- data_prep_gain(sc_data_list[[6]])[[1]] )[3]
)

### SI
data_prep_si <- function(data_in)
{
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  return(si_model(data_train, data_test))
}

si_time <- c(
  system.time(temp <- data_prep_si(sc_data_list[[1]])[[1]]  )[3],
  system.time(temp  <- data_prep_si(sc_data_list[[2]])[[1]] )[3],
  system.time(temp  <- data_prep_si(sc_data_list[[3]])[[1]] )[3],
  system.time(temp  <- data_prep_si(sc_data_list[[4]])[[1]] )[3],
  system.time(temp  <- data_prep_si(sc_data_list[[5]])[[1]] )[3],
  system.time(temp  <- data_prep_si(sc_data_list[[6]])[[1]] )[3]
)

## ReFE
data_prep_refe <- function(data_in)
{
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  return(refe_model(data_train, data_test))
}

refe_time <- c(
  system.time(refe_pred[[1]] <- data_prep_refe(sc_data_list[[1]])[[1]]  )[3],
  system.time(refe_pred[[2]]  <- data_prep_refe(sc_data_list[[2]])[[1]] )[3],
  system.time(refe_pred[[3]]  <- data_prep_refe(sc_data_list[[3]])[[1]] )[3],
  system.time(refe_pred[[4]]  <- data_prep_refe(sc_data_list[[4]])[[1]] )[3],
  system.time(refe_pred[[5]]  <- data_prep_refe(sc_data_list[[5]])[[1]] )[3],
  system.time(refe_pred[[6]]  <- data_prep_refe(sc_data_list[[6]])[[1]] )[3]
)

### MI 
data_prep_mi <- function(data_in)
{
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  return(mi_model(data_train, data_test))
}

mi_time <- c(
   system.time(mi_pred[[1]] <- data_prep_mi(sc_data_list[[1]])[[1]]  )[3],
   system.time(mi_pred[[2]]  <- data_prep_mi(sc_data_list[[2]])[[1]] )[3],
   system.time(mi_pred[[3]]  <- data_prep_mi(sc_data_list[[3]])[[1]] )[3],
   system.time(mi_pred[[4]]  <- data_prep_mi(sc_data_list[[4]])[[1]] )[3],
   system.time(mi_pred[[5]]  <- data_prep_mi(sc_data_list[[4]])[[1]] )[3]   
 )
 ## 1 mill takes too much time. Even 50000 takes more than a day to run
 mi_time[[6]] <- NA

## ReFE
data_prep_lazy <- function(data_in)
{
  cu = round(3*nrow(data_in)/4)
  data_train = data_in[1:cu,]
  data_test = data_in[(cu+1):nrow(data_in),]
  data_train_y <- data.frame(data_train[,outcome])
  data_train_x <- data_train[,!names(data_train) %in% c(outcome)]
  names(data_train_y) = outcome
  return(lazy_reg_model(data_train_x,data_train_y, data_test, type="anova"))
}

lazy_time <- c(
  system.time(lazy_pred[[1]] <- data_prep_lazy(sc_data_list[[1]])[[1]]  )[3],
  system.time(lazy_pred[[2]]  <- data_prep_lazy(sc_data_list[[2]])[[1]] )[3],
  system.time(lazy_pred[[3]]  <- data_prep_lazy(sc_data_list[[3]])[[1]] )[3]
)
## 500,000 and 1 mill takes more than 24 hours to run. Hence we do not run these simulations for lazy method in the reproducibilit code
lazy_time[[4]] <- NA
lazy_time[[5]] <- NA
lazy_time[[6]] <- NA

### OTHER METHODS not reported in paper:
# ###  DISCOM
# data_prep_discom <- function(data_in)
# {
#   data_train_y <- data.frame(data_in[,outcome])
#   data_train_x <- data_in[,!names(data_in) %in% c(outcome)]
#   names(data_train_y) = outcome
#   data_test = data_in[1:3,]
#   return(discom_model(4, data_in, data_test, low_threshold = 0.05))
# }
# 
# discom_time <- c(
#   system.time(temp <- data_prep_discom(sc_data_list[[1]])  )[3],
#   system.time(temp  <- data_prep_discom(sc_data_list[[2]]) )[3],
#   system.time(temp  <- data_prep_discom(sc_data_list[[3]])[[1]] )[3],
#   system.time(temp  <- data_prep_discom(sc_data_list[[4]])[[1]] )[3],
#   system.time(temp  <- data_prep_discom(sc_data_list[[5]])[[1]] )[3],
#   system.time(temp  <- data_prep_discom(sc_data_list[[6]])[[1]] )[3],
#   system.time(temp  <- data_prep_discom(sc_data_list[[7]])[[1]] )[3]
# )
# ### iMSF
# data_prep_imsf <- function(data_in)
# {
#   data_train_y <- data.frame(data_in[,outcome])
#   data_train_x <- data_in[,!names(data_in) %in% c(outcome)]
#   names(data_train_y) = outcome
#   data_test = data_in[1:3,]
#   return(imsf_model(4, data_in, data_test, low_threshold = 0.05))
# }
# 
# imsf_time <- c(
#   system.time(temp <- data_prep_imsf(sc_data_list[[1]])[[1]]  )[3],
#   system.time(temp  <- data_prep_imsf(sc_data_list[[2]])[[1]] )[3],
#   system.time(temp  <- data_prep_imsf(sc_data_list[[3]])[[1]] )[3],
#   system.time(temp  <- data_prep_imsf(sc_data_list[[4]])[[1]] )[3],
#   system.time(temp  <- data_prep_imsf(sc_data_list[[5]])[[1]] )[3],
#   system.time(temp  <- data_prep_imsf(sc_data_list[[6]])[[1]] )[3],
#   system.time(temp  <- data_prep_imsf(sc_data_list[[7]])[[1]] )[3]
# )

# ## CART
# data_prep_cart <- function(data_in)
# {
#   data_train_y <- data.frame(data_in[,outcome])
#   data_train_x <- data_in[,!names(data_in) %in% c(outcome)]
#   names(data_train_y) = outcome
#   data_test = data_in[1:3,]
#   return(cart_model(data_in, data_test, type="anova"))
# }
# 
# cart_time <- c(
#   system.time(temp <- data_prep_cart(sc_data_list[[1]])[[1]]  )[3],
#   system.time(temp  <- data_prep_cart(sc_data_list[[2]])[[1]] )[3],
#   system.time(temp  <- data_prep_cart(sc_data_list[[3]])[[1]] )[3],
#   system.time(temp  <- data_prep_cart(sc_data_list[[4]])[[1]] )[3],
#   system.time(temp  <- data_prep_cart(sc_data_list[[5]])[[1]] )[3],
#   system.time(temp  <- data_prep_cart(sc_data_list[[6]])[[1]] )[3],
#   system.time(temp  <- data_prep_cart(sc_data_list[[7]])[[1]] )[3]
# )
# 

scalability = data.frame(
brm_time ,
brm_n_ov_time,
listwise_time ,
columnwise_time ,
mi_time ,
si_time ,
refe_time ,
gain_time ,
lazy_time
)

write.csv(scalability,"/root/capsule/results/Scalability_reg_bike.csv")
