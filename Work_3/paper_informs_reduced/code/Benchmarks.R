listwise_model <- function(data_train, data_test)
{
  data_train_listwise <- na.omit(data_train)
  reg_listwise <- reg_train(data_train_listwise)
  tree_listwise <- tree_train(data_train_listwise)
  data_test_imputed <- impute_test(data_train,data_test)
  listwise_pred_reg <- pred_fun(reg_listwise, data_test_imputed)
  listwise_pred_tree <- pred_fun(tree_listwise, data_test_imputed )
  return(list(listwise_pred_reg, listwise_pred_tree))
}

columnwise_model <- function(data_train, data_test)
{
 miss_prop_cols <- sapply(data_train, function(y) sum(length(which(is.na(y)))))
  miss_proportion <- miss_prop_cols/(dim(data_train)[1])
  keep_list <- which(miss_proportion <= 0.4)
  data_train_columnwise <- data_train[,keep_list]
  data_train_columnwise <- impute_manual(data_train_columnwise)
  reg_columnwise <- reg_train(data_train_columnwise)
  tree_columnwise <- tree_train(data_train_columnwise)
  data_test_imputed <- impute_test(data_train,data_test)
  columnwise_pred_reg <- pred_fun(reg_columnwise, data_test_imputed)
  columnwise_pred_tree <- pred_fun(tree_columnwise, data_test_imputed )
  return(list(columnwise_pred_reg, columnwise_pred_tree))
}

mi_model <- function(data_train,data_test)
{
  
  data_test_imputed_mi <-  impute_test(data_train,data_test)  
  data_mi <- mice(data_train,m=5,maxit=3,meth='pmm',seed=1234) 
  completedData <- mice::complete(data_mi,1)
  model_mi_tree <- list()
  model_mi_reg <- list()
  
  for(list in 1:data_mi$m)
  {
    try( model_mi_tree[[list]] <-  tree_train( mice::complete(data_mi,list)  ) )
    
    try( model_mi_reg[[list]] <-   reg_train( mice::complete(data_mi,list)  ) )
  }
  model_meta_var_tree <- data.frame(matrix(NA, nrow = nrow(completedData), ncol = data_mi$m ))
  model_meta_var_reg <-  data.frame(matrix(NA, nrow = nrow(completedData), ncol = data_mi$m ))
  for(i in 1:data_mi$m)
  {
    try( model_meta_var_tree_sc <- pred_fun(model_mi_tree[[i]], mice::complete(data_mi,list) ) ) 
    try( model_meta_var_reg_sc <-  pred_fun(model_mi_reg[[i]],  mice::complete(data_mi,list) )) 
    
    model_meta_var_tree[,i] <- model_meta_var_tree_sc 
    model_meta_var_tree[is.na(model_meta_var_tree[,i]),i] = 0
    
    model_meta_var_reg[,i] = model_meta_var_reg_sc
    model_meta_var_reg[is.na(model_meta_var_reg[,i]),i] = 0
  }
  
  Y = data_train[,names(data_train) %in% names(data_train_y) ] 
  
  Xrf <- model_meta_var_tree
  datam1 <- data.table(Xrf,Y)
  
  rf.model_meta_tree <- randomForest(Y ~.,data=datam1,importance =FALSE,ntree=50)
  
  Xreg <- model_meta_var_reg
  datam2 <- data.table(Xreg,Y)
  
  rf.model_meta_reg <- randomForest(Y~.,data=datam2,importance =FALSE,ntree=50)
  
  ## Prediction is same procedure
  meta_var_pred_tree <- data.frame(matrix(NA, nrow = nrow(data_test), ncol = data_mi$m)) 
  meta_var_pred_reg <- data.frame(matrix(NA, nrow = nrow(data_test), ncol = data_mi$m)) 
  for(i in 1:data_mi$m)
  {
    try( meta_var_pred_tree[,i] <- pred_fun(model_mi_tree[[i]],data_test_imputed_mi)  )
    try( meta_var_pred_reg[,i] <- pred_fun(model_mi_reg[[i]],data_test_imputed_mi) )
  }
  
  tree_meta_pred_tree <- predict(rf.model_meta_tree,meta_var_pred_tree)
  tree_meta_pred_reg <- predict(rf.model_meta_reg,meta_var_pred_reg)
  return(list(tree_meta_pred_reg,tree_meta_pred_tree))
}


### SoftImpute = SI:
si_model <- function(data_train,data_test)
{
  X_in <- data_train[,names(data_train) %in% names(data_train_x) ]
  Y_in <- data.frame(data_train[,names(data_train) %in% names(data_train_y) ])
  
  X_out <- data_test[,names(data_test) %in% names(data_train_x) ]
  Y_out <- data.frame(data_test[,names(data_test) %in% names(data_train_y) ])
  
  names(Y_in) = names(data_train_y)
  names(Y_out) = names(data_train_y)
  
  data_ohc <- makeX(train = X_in,na.impute= FALSE, test = X_out) 
  data_train_ohc = data_ohc[[1]]
  data_test_ohc = data_ohc[[2]]
  
  si.fit=softImpute(data_train_ohc,rank=round(ncol(data_train_ohc)/2),lambda=30)
  si.data=softImpute::complete(data_train_ohc,si.fit)
  si.test = softImpute::complete(data_test_ohc,si.fit)
  
  si.train = cbind(si.data,Y_in)
  si.test = cbind(si.test,Y_out)
  
  si.model_tree <- tree_train(si.train)
  si.pred_tree <- pred_fun(si.model_tree, si.test)
  si.model_reg <- reg_train(si.train) 
  si.pred_reg <- pred_fun(si.model_reg, si.test)
  
  return(list(si.pred_reg, si.pred_tree))
}

### Mean Impute:
mean_model <- function(data_train,data_test)
{
  data_test_imputed <-  impute_test(data_train,data_test) 
  data_train_imputed <-  impute_manual(data_train) 
  mean.model_tree <- tree_train(data_train_imputed) 
  mean.pred_tree <- pred_fun(mean.model_tree, data_test_imputed)
  mean.model_reg <- reg_train(data_train_imputed)
  mean.pred_reg <- pred_fun(mean.model_reg, data_test_imputed)
  return(list(mean.pred_reg, mean.pred_tree))
}

lazy_tree_model <- function(X_in, Y_in,data_test_l, type="class")
{ 
  #data_test_l = data_test[1:100,]
  #Y_in = data_train_y
  #X_in = data_train_x 
  t_data_present <- data.frame( matrix(as.integer(!is.na(data_test_l)),nrow = nrow(data_test_l), ncol = ncol(data_test_l))) 
  names(t_data_present) <- names(data_test_l) 
  t_data_present <- t_data_present[,!names(t_data_present) %in% names(Y_in)]
  
  data_test_l$treepred <- NA
  data_test_l$regpred  <- NA
  
  ## Simple row by row iteration:
  for (i in 1:nrow(t_data_present))
  {
    X_in_c = X_in[,as.logical(t_data_present[i,])]
    data_tr_lazy <- cbind(X_in_c, Y_in)
    data_tr_lazy = na.omit(data_tr_lazy)
    #lazy_tree_m = tree_train(data_tr_lazy)

    ## Simple linear regression works 150 times faster than stepwise procedure
    X <- data_tr_lazy[,names(data_tr_lazy) %in% names(X_in) ]
    Y <- data_tr_lazy[,names(data_tr_lazy) %in% names(Y_in) ]
    datam <- data.table(X,Y)

    fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))

    #lazy_reg_m <- lm( fmla, data=data_tr_lazy)
    try( lazy_reg_m <- reg_train(data_tr_lazy) )

    try( lazy_tree_m <- rpart(fmla, method=type, data=datam ) )
	
    try( data_test_l$treepred[i] <- pred_fun(lazy_tree_m, data_test_l[i,], type=type) )
    try( data_test_l$regpred[i] <- pred_fun(lazy_reg_m, data_test_l[i,]) )
  }
  return(list(data_test_l$regpred, data_test_l$treepred))
  #return(data_test_l$regpred)
}


lazy_tree_only_model <- function(X_in, Y_in,data_test_l, type="class")
{ 
  #data_test_l = data_test[1:100,]
  #Y_in = data_train_y
  #X_in = data_train_x 
  t_data_present <- data.frame( matrix(as.integer(!is.na(data_test_l)),nrow = nrow(data_test_l), ncol = ncol(data_test_l))) 
  names(t_data_present) <- names(data_test_l) 
  t_data_present <- t_data_present[,!names(t_data_present) %in% names(Y_in)]
  
  data_test_l$treepred <- NA
  data_test_l$regpred  <- NA
  
  ## Simple row by row iteration:
  for (i in 1:nrow(t_data_present))
  {
    X_in_c = X_in[,as.logical(t_data_present[i,])]
    data_tr_lazy <- cbind(X_in_c, Y_in)
    data_tr_lazy = na.omit(data_tr_lazy)
    #lazy_tree_m = tree_train(data_tr_lazy)

    ## Simple linear regression works 150 times faster than stepwise procedure
    X <- data_tr_lazy[,names(data_tr_lazy) %in% names(X_in) ]
    Y <- data_tr_lazy[,names(data_tr_lazy) %in% names(Y_in) ]
    datam <- data.table(X,Y)

    fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))


    lazy_tree_m <- rpart(fmla, method=type, data=datam ) 
	
    data_test_l$treepred[i] = pred_fun(lazy_tree_m, data_test_l[i,], type=type) 
  }
  return(data_test_l$treepred)
}

lazy_reg_model <- function(X_in, Y_in,data_test_l, type="class")
{ 
  #data_test_l = data_test[1:100,]
  #Y_in = data_train_y
  #X_in = data_train_x 
  t_data_present <- data.frame( matrix(as.integer(!is.na(data_test_l)),nrow = nrow(data_test_l), ncol = ncol(data_test_l))) 
  names(t_data_present) <- names(data_test_l) 
  t_data_present <- t_data_present[,!names(t_data_present) %in% names(Y_in)]
  
  data_test_l$treepred <- NA
  data_test_l$regpred  <- NA
  
  ## Simple row by row iteration:
  for (i in 1:nrow(t_data_present))
  {
    X_in_c = X_in[,as.logical(t_data_present[i,])]
    data_tr_lazy <- cbind(X_in_c, Y_in)
    data_tr_lazy = na.omit(data_tr_lazy)
    #lazy_tree_m = tree_train(data_tr_lazy)

    ## Simple linear regression works 150 times faster than stepwise procedure
    X <- data_tr_lazy[,names(data_tr_lazy) %in% names(X_in) ]
    Y <- data_tr_lazy[,names(data_tr_lazy) %in% names(Y_in) ]
    datam <- data.table(X,Y)

    fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))

    #lazy_reg_m <- lm( fmla, data=data_tr_lazy)
    lazy_reg_m = reg_train(data_tr_lazy)

    data_test_l$regpred[i] = pred_fun(lazy_reg_m, data_test_l[i,])
  }
  return(data_test_l$regpred)
}

### CART:
cart_model <- function(data_train, data_test, type="class")
{
  X <- data_train[,!names(data_train) %in% names(data_train_y) ]
  Y <- data_train[,names(data_train) %in% names(data_train_y) ]
  #Y = as.factor(Y)
  X <- remove_constant(X, na.rm= FALSE, quiet = TRUE)
  datam <- data.table(X,Y) 
  fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))
  # cart_m <- rpart(fmla,          method=type, data=data_train )
  cart_m <- rpart(fmla, method=type, data=datam )
  cart_prediction <- predict(cart_m, data_test)
  if(type == "anova") { return(cart_prediction) }
  else{  return(cart_prediction[,2] ) }
}

### CART with indicator:
cart_i_model <- function(data_train, data_test, type="class")
{ 
  X_tr <- data_train[,!names(data_train) %in% names(data_train_y) ]
  Y_tr <- data_train[,names(data_train) %in% names(data_train_y) ]

  X_test <- data_test[,!names(data_test) %in% names(data_train_y) ]
  Y_test <- data_test[,names(data_test) %in% names(data_train_y) ]
  
  tr_data_present <- data.frame( matrix(as.integer(!is.na(X_tr)),nrow = nrow(X_tr), ncol = ncol(X_tr))) 
  data_train_aug <- cbind(data_train,tr_data_present)
  
  test_data_present <- data.frame( matrix(as.integer(!is.na(X_test)),nrow = nrow(X_test), ncol = ncol(X_test))) 
  data_test_aug <- cbind(data_test,test_data_present)

  X <- data_train_aug[,!names(data_train_aug) %in% names(data_train_y) ]
  Y <- data_train_aug[,names(data_train_aug) %in% names(data_train_y) ]
  #Y = as.factor(Y)
  X <- remove_constant(X, na.rm= FALSE, quiet = TRUE)
  datam <- data.table(X,Y) 
  fmla <- as.formula(paste("Y ~ ", paste(names(X), collapse= "+")))
  # cart_m <- rpart(fmla,          method=type, data=data_train )
  cart_m <- rpart(fmla, method=type, data=datam )
  cart_prediction <- predict(cart_m, data_test_aug)
  if(type == "anova") { return(cart_prediction) }
  else{  return(cart_prediction[,2] ) }
}

refe_model <- function(data_train,data_test)
{
  train_sets = replicate((ncol(data_train)-1),data.frame())
  mvi_sets = replicate((ncol(data_train)-1),rep(1,(ncol(data_train)-1))) ## Indicator matrix 
  
  ### Re-arrange the data_train and data_test to have outcome as last column
  X_in <- data_train[,names(data_train) %in% names(data_train_x) ]
  Y_in <- data.frame(data_train[,names(data_train) %in% names(data_train_y) ])
  
  X_out <- data_test[,names(data_test) %in% names(data_train_x) ]
  Y_out <- data.frame(data_test[,names(data_test) %in% names(data_train_y) ])
  
  names(Y_in) = names(data_train_y)
  names(Y_out) = names(data_train_y)
  
  data_train = cbind(X_in,Y_in)
  data_test = cbind(X_out,Y_out)
  
  
  for (col in 1:(ncol(data_train)-1))
  {
    train_sets[[col]] = data_train[,-col]
    train_sets[[col]] = impute_manual(train_sets[[col]])
    mvi_sets[col,col] <- 0
  }
  
  reg_m <- lapply(train_sets,reg_train)
  tree_m <- lapply(train_sets,tree_train)
  
  ## pred_reduced is a BRM function that looks at closest data subset to match each test instance to make the predictions 
  pred_reg_outcomes  <- pred_reduced(reg_m,  data_test,data_train_x, data_train_y,mvi_sets)
  pred_tree_outcomes <- pred_reduced(tree_m, data_test,data_train_x, data_train_y,mvi_sets)
  # data_test_imputed <-  impute_test(data_train,data_test)  
  return(list(pred_reg_outcomes,pred_tree_outcomes))
}

### iMSF: (from Yu 2020 code):
imsf_model <- function(num_blocks, data_train,data_test,low_threshold = 0.05, bin_indicator=0)
{
  X_in <- data_train[,names(data_train) %in% names(data_train_x) ]
  Y_in <- data_train[,names(data_train) %in% names(data_train_y) ]
  
  X_out <- data_test[,names(data_test) %in% names(data_train_x) ]
  Y_out <- data_test[,names(data_test) %in% names(data_train_y) ]
  
  X_in_n_ind <- sapply(X_in, is.numeric)
  X_in[X_in_n_ind] <- lapply(X_in[X_in_n_ind], scale)
  
  X_out_n_ind <- sapply(X_out, is.numeric)
  X_out[X_out_n_ind] <- lapply(X_out[X_out_n_ind], scale)
  
  data_ohc <- makeX(train = X_in,na.impute= FALSE, test = X_out) 
  data_train_ohc = data_ohc[[1]]
  data_test_ohc = data_ohc[[2]]
  
  X_in = data.frame(data_train_ohc)
  X_out = data.frame(data_test_ohc)
  
  ## Set aside data for tuning:
  X_in_tr = X_in[1:round(0.75*nrow(X_in)),]
  X_in_tun = X_in[(round(0.75*nrow(X_in))+1):nrow(X_in),]
  Y_in_tr = data.frame(Y_in[1:round(0.75*length(Y_in))])
  Y_in_tun = data.frame(Y_in[(round(0.75*length(Y_in))+1):length(Y_in)])
  
  ## Tuning and test do not have missing values (as per Yu 2020):
  X_in_tun = impute_manual(X_in_tun)
  X_out = impute_manual(X_out)
  
  ## Column sets are gathered using brm_ov_subset code
  brm = brm_ov_subsets(num_blocks,X_in_tr, Y_in_tr,low_threshold)
  column_set = brm[[4]]
  n_ov_subsets = brm[[3]]
  
  ## Create the y.new and x.new:
  Y_in.new = Y_in_tr
  k=1
  for (i in 1:length(n_ov_subsets))
  {
    n = nrow(n_ov_subsets[[i]])
    Y_in.new[k:(n+k-1),1] = Y_in_tr[k:(n+k-1),1]/sqrt(n)
    k = k+n
  }
    
  ### Weight each block 
  n_ov_subsets_w = lapply(n_ov_subsets,function(x) as.matrix(x[,1:(ncol(x)-1)]/sqrt(dim(x)[1]) ))
  
  ## Block diagonal matrix of all the blocks:
  bdiag_ = do.call(bdiag,  n_ov_subsets_w)
  X.train.new = as.matrix(bdiag_)
  
  ###
  Y.train.new = Y_in.new[,1]
  column_set_x = lapply(column_set, function(x) x[1:(length(x)-1)] )
  group_index_name = unlist(column_set_x)
  
  ## We assume that the alphabetic order of X_in_tun columns should be same as column_set_x
  group.index = as.numeric(factor(group_index_name))  ## Number in alphabetic order of columns
  
  col.index = as.numeric(factor(names(X_in_tun)))  ## Number in alphabetic order of columns
  
  ### Get the first columns in group.index with the index of each col.index for beta mapping 
  ### back to names(X_in_tun)
  ## https://stackoverflow.com/questions/46716645/obtaining-the-first-instance-of-every-element-in-a-list-using-r
  index_X_tun <- unlist(purrr::map(col.index, function(i){
    min(which(group.index == i))
  }))
  
  X.tun = as.matrix(X_in_tun)
  X.test = as.matrix(X_out)
  Y.tun = Y_in_tun[,1]
  nlambda=5
  lambda.all=seq(1,0.05,length=nlambda)
  imsf.tun.error=rep(NA,nlambda)
  
  for(i in 1:nlambda){
    imsf=grplasso(X.train.new,Y.train.new,group.index,lambda=lambda.all[i],center=F,
                  standardize=F,model=LinReg(),
                  control = grpl.control(trace = 0))
    beta.imsf=as.vector(imsf$coefficients[index_X_tun])
    imsf.tun.values=as.vector(X.tun%*%beta.imsf)
    imsf.tun.error[i]=mean((Y.tun-imsf.tun.values)^2)
  }
  opt.lambda=lambda.all[which.min(imsf.tun.error)]
  imsf.fit=grplasso(X.train.new,Y.train.new,group.index,lambda=opt.lambda,center=F,
                    standardize=F,model=LinReg(),
                    control = grpl.control(trace = 0))
  
  # pred_imsf = predict(imsf.fit, newdata = X_out)
  beta.imsf=as.vector(imsf.fit$coefficients[index_X_tun])
  
  var_selected = which(beta.imsf  != 0 )
  X_feat = data.frame(X_in[,var_selected])
  names(X_feat) = names(X_in)[var_selected]
  
  # pred_imsf=as.vector(X.test%*%beta.imsf)
  datam <- data.table(X_feat,Y_in)
  fmla <- as.formula(paste("Y_in ~ ", paste(names(X_feat), collapse= "+")))
  if( bin_indicator==1)
  {
  model_imsf <- glm( fmla, data=datam,family="binomial")
  pred_imsf = predict(model_imsf,X_out,type= "response")
  }
  else
  {
  model_imsf <- lm( fmla, data=datam)
  pred_imsf = predict(model_imsf,X_out)
  }
  return(pred_imsf)
}

#### DISCOM - FAST
discom_model <- function(num_blocks, data_train,data_test,low_threshold = 0.05, bin_indicator=0)
{
  compute.xtx<-function(x,robust=0,k_value=1) 
    # robust=1 for huber estimate, k_value is used in huber function
  {
    p=ncol(x)
    cov.matrix=matrix(NA,p,p)
    if(robust==0){cov.matrix=cov(x,use="pairwise.complete.obs")}
    else{
      for(i in 1:p)
      {
        for(j in 1:p){
          index=which((!is.na(x[,i]))&(!is.na(x[,j])))
          x1=x[index,i]
          x2=x[index,j]
          cov.matrix[i,j]=huberM((x1-mean(x1))*(x2-mean(x2)),k=k_value*sqrt(length(index)/log(p)))$mu
        }
      }
    }
    cov.matrix  
  }
  
  compute.xty<-function(x,y,robust=0,k_value=1)
    # robust=1 for huber estimate, k_value is used in huber function
  {
    p=ncol(x)
    cov.vector=rep(NA,p)
    if(robust==0){cov.vector=cov(x,y,use="pairwise.complete.obs")}
    else{
      for(i in 1:p){
        index=which((!is.na(x[,i])))
        x1=x[index,i]
        x2=y[index]
        cov.vector[i]=huberM((x1-mean(x1))*(x2-mean(x2)),k=k_value*sqrt(length(index)/log(p)))$mu
      }
    }
    cov.vector
  }
  
  ### We assume that the blocks are identified using our method 
  X_in <- data_train[,names(data_train) %in% names(data_train_x) ]
  Y_in <- data_train[,names(data_train) %in% names(data_train_y) ]
  
  X_out <- data_test[,names(data_test) %in% names(data_train_x) ]
  Y_out <- data_test[,names(data_test) %in% names(data_train_y) ]
  
  X_in_n_ind <- sapply(X_in, is.numeric)
  X_in[X_in_n_ind] <- lapply(X_in[X_in_n_ind], scale)
  
  X_out_n_ind <- sapply(X_out, is.numeric)
  X_out[X_out_n_ind] <- lapply(X_out[X_out_n_ind], scale)
   
  ## If variable is numeric, we scale it, in the input matrix, 
  ### for ease of computation
  data_ohc <- makeX(train = X_in,na.impute= FALSE, test = X_out) 
  data_train_ohc = data_ohc[[1]]
  data_test_ohc = data_ohc[[2]]
  
  X_in = data.frame(data_train_ohc)
  X_out = data.frame(data_test_ohc)
  
  ## Set aside data for tuning:
  X_in_tr = X_in[1:round(0.75*nrow(X_in)),]
  X_in_tun = X_in[(round(0.75*nrow(X_in))+1):nrow(X_in),]
  Y_in_tr = data.frame(Y_in[1:round(0.75*length(Y_in))])
  Y_in_tun = data.frame(Y_in[(round(0.75*length(Y_in))+1):length(Y_in)])
  
  ## Remove columns with all null values:   X_in_tr <- X_in_tr[,colSums(is.na(X_in_tr))<nrow(X_in_tr)]  X_in_tun <- X_in_tun[,names(X_in_tun) %in% names(X_in_tr)]

  ## Tuning and test do not have missing values (as per Yu 2020):
  X_in_tun = impute_manual(X_in_tun)
  X_out = impute_manual(X_out)
  
  ## Column sets are gathered using brm_ov_subset code
  brm = brm_ov_subsets(num_blocks,X_in_tr, Y_in_tr,low_threshold)
  column_set = brm[[4]]
  n_ov_subsets = brm[[3]]
  column_set_x = lapply(column_set, function(x) x[1:(length(x)-1)] )
  
  nalpha=10
  nlambda=10    
  alpha.all=10^seq(0,-3,length=nalpha)
  lambda.all=10^(seq(log10(2),-3,length=nlambda))
  p = ncol(X_in_tr)
  
  ## Retain only unique elements in each list to get grouping of columns
  
  column_set_x2 = column_set_x[order(sapply(column_set_x, length))]
  
    ### Unnecessary code. Updated with two lines of code for the necessary column ordering    # col_groups <- list()    # i = 1    # ### Set theory style (used for creating overlapping subsets earlier):    # for(list1 in 1:length(column_set_x))    # {    #   for(list2 in 1:length(column_set_x))    #   {    #     if( length(column_set_x[[list2]]) > length(column_set_x[[list1]])  )    #     {    #       subset_cols <- column_set_x2[[list2]] [(column_set_x2[[list2]]  %in% column_set_x2[[list1]])]    #       col_groups[[i]] = subset_cols    #       i = i + 1    #     }    #   }    # }    #     # col_groups = col_groups[lengths(col_groups) != 0]    # col_groups = col_groups[!duplicated(col_groups) ]    #     # ## col_groups = col_groups[order(sapply(col_groups, length))] ## Already done earlier      #     # ### https://stackoverflow.com/questions/45318034/remove-duplicated-elements-from-list    # # make a copy of my.list    # res.col_groups <- col_groups    # # take set difference between contents of list elements and accumulated elements    # res.col_groups[-1] <- mapply("setdiff", res.col_groups[-1],    #                              head(Reduce(c, col_groups, accumulate=TRUE), -1))    #     # # res.col_groups[-1] <- mapply("setdiff", res.col_groups[-1],    # #                             head(Reduce( function(x, y) unique(c(x, y)) , col_groups, accumulate=TRUE), -1)) 
  ## These are exclusive columns that do not appear in other groups. 
  #column_set_c = unlist(res.col_groups)
   ## Update in Revision 2, this code avoids the errors encountered in previous version.   column_set_x2[-1] <- mapply("setdiff", column_set_x2[-1],                              head(Reduce(c, column_set_x2, accumulate=TRUE), -1))    res.col_groups = append(column_set_x2, list(setdiff(names(X_in_tr), unlist(column_set_x2))) )  res.col_groups = res.col_groups[lengths(res.col_groups) != 0]	  
  #if(length(unlist(res.col_groups)) != ncol(X_in_tr))
  #{
  #  print("Error in finding column blocks")
  #}

  xtx.raw=compute.xtx(X_in_tr)
  xty=compute.xty(X_in_tr,Y_in_tr)
  
  rownames(xtx.raw) = names(X_in_tr)
  colnames(xtx.raw) = names(X_in_tr)
  
  ## Calculate that many cov as there are exclusive column groups (i.e., column blocks):
  xtx_col = lapply(res.col_groups, function(x) xtx.raw[x,x] ) 
  
  ## The covariance mat as block-diagonal 
  xtx.raw.I_ = do.call(bdiag,  xtx_col)
  xtx.raw.I = as.matrix(xtx.raw.I_)
  
  xtx.raw.C=xtx.raw-xtx.raw.I
  shrink.target=sum(diag(xtx.raw))/p
  
  X.tun = as.matrix(X_in_tun)
  Y.tun = Y_in_tun[,1]
  X.test = as.matrix( X_out )
  
  c1=sqrt(log(p)/200)
  c2=sqrt(log(p)/100)
  A.matrix=(c2-c1)*xtx.raw.I+c1*shrink.target*diag(p)
  K.min=max(0,min(eigen(xtx.raw)$values)/(c2*min(eigen(xtx.raw)$values)-min(eigen(A.matrix)$values)))
  K.max=1/c2
  n.K=30  
  nlambda=30
  K.all=seq(K.min,K.max,length=n.K)
  lambda.all=10^(seq(log10(2),-3,length=nlambda))
  DISCOM.fast.tun.error=matrix(NA,n.K,nlambda)
  
  for(i in 1:n.K){
    alpha1=1-K.all[i]*c1
    alpha2=1-K.all[i]*c2
    xtx=alpha1*xtx.raw.I+alpha2*xtx.raw.C+(1-alpha1)*shrink.target*diag(p)
    beta.initial=rep(0,p)
    for(k in 1:nlambda){
      beta.cov=as.vector(crossProdLasso(xtx,xty,lambda.all[k],beta.init=beta.initial)$beta)
      betal.initial=beta.cov
      DISCOM.fast.tun.values=as.vector(as.matrix(X.tun)%*%beta.cov)
      DISCOM.fast.tun.error[i,k]=mean((Y.tun-DISCOM.fast.tun.values)^2)
    }
  }
  
  opt.index=as.vector(which(DISCOM.fast.tun.error==min(DISCOM.fast.tun.error),arr.ind=T)[1,])
  opt.alpha1=1-K.all[opt.index[1]]*c1
  opt.alpha2=1-K.all[opt.index[1]]*c2
  opt.lambda=lambda.all[opt.index[2]]
  xtx=opt.alpha1*xtx.raw.I+opt.alpha2*xtx.raw.C+(1-opt.alpha1)*shrink.target*diag(p)
  # beta.cov=as.vector(crossProdLasso(xtx,xty,opt.lambda)$beta)
  #y_pred=as.vector(X.test%*%beta.cov)
  beta.cov=as.vector(crossProdLasso(xtx.raw,xty,opt.lambda)$beta)
 
  var_selected = which(beta.cov  != 0 )
  X_feat = data.frame(X_in[,var_selected])
  names(X_feat) = names(X_in)[var_selected]
  datam <- data.table(X_feat,Y_in)
  fmla <- as.formula(paste("Y_in ~ ", paste(names(X_feat), collapse= "+")))

    if( bin_indicator==1)
    {
    model_discom <- glm( fmla, data=datam,family="binomial")
    pred_discom = predict(model_discom,X_out,type= "response")
    }
    else
    {
    model_discom <- lm( fmla, data=datam)
    pred_discom = predict(model_discom,X_out)
    } 
 
  return(pred_discom)
}

gain_model <- function(data_train,data_test)
{
  gain_parameters = c(batch_size= 10, hint_rate= 0.01, alpha= 0.5, iterations= 10)
  
  X_in <- data_train[,names(data_train) %in% names(data_train_x) ]
  Y_in <- data_train[,names(data_train) %in% names(data_train_y) ]
  
  X_out <- data_test[,names(data_test) %in% names(data_train_x) ]
  Y_out <- data_test[,names(data_test) %in% names(data_train_y) ]
  
  data_ohc <- makeX(train = X_in,na.impute= FALSE, test = X_out) 
  data_train_ohc = data_ohc[[1]]
  data_test_ohc = data_ohc[[2]]
  
  X_in = data_train_ohc
  X_out = data_test_ohc
  
  data_test_imputed <- impute_test(data_train,data_test)
  data_train_imputed <- gain$gain(X_in)
  data_train_imputed = data.frame(data_train_imputed)
  X_in = data.frame(X_in)
  names(data_train_imputed) = names(X_in)
  data_train_imputed = cbind(data_train_imputed,Y_in)
  names(data_train_imputed)[ncol(data_train_imputed)] = names(data_train_y)
  
  gain.model_tree <- tree_train( data_train_imputed) 
  gain.pred_tree <- pred_fun(gain.model_tree ,data_test_imputed)
  
  gain.model_reg <- reg_train(data_train_imputed) 
  gain.pred_reg <- pred_fun(gain.model_reg, data_test_imputed)
  return(list(gain.pred_reg, gain.pred_tree))
}