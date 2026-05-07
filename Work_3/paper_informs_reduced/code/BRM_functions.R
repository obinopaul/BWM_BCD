## For faster execution
brm_num_blocks <- function(data_train_x,low_threshold = 0.05, n=10)
{

  ## For larger datasets, we consider representative samples: 
  if(nrow(data_train_x) > 50000)
  {
     set.seed(12345)
     data_train_x = data_train_x[sample(nrow(data_train_x),50000,replace=FALSE ),]
  }

  data_present <- data.frame( matrix(as.integer(!is.na(data_train_x)),nrow = nrow(data_train_x), ncol = ncol(data_train_x))) 
  names(data_present) <- names(data_train_x) 
  
  missing_prop <- function(MVI,k, low_threshold, n=10)
  {
    determine_best_cluster <- function(data_in,clus_number, n=10)
    {
      # n <- 10
      kmeans_o <- lapply(1:n,function(x) { set.seed(x); a<-kmeans(data_in,clus_number); return(a) } ) 
      opt_kmeans <- which.max(lapply(kmeans_o,function(x) {100-x$tot.withinss/x$totss*100})) 
      best_cluster <- kmeans_o[[opt_kmeans]]
      return(best_cluster)
    }	
    
    member_clusters <- determine_best_cluster(MVI,k)
    members <- member_clusters$cluster
    data_subs_1 <- split(MVI,members)  
    data_subs_2 <- list(length(data_subs_1)) ## This is for identifying the columns
    missingness <- vector(length = length(data_subs_1))
    for(list in 1:length(data_subs_1))
    {
	data_subs_1[[list]] = data.frame(data_subs_1[[list]])
      completeness <- colSums(data_subs_1[[list]])
      data_subs_2[[list]] <- data_subs_1[[list]][,which(completeness > (low_threshold)*nrow(data_subs_1[[list]])) ]  
	  data_subs_2[[list]] = data.frame(data_subs_2[[list]])
	  if(ncol(data_subs_2[[list]]) > 1)
	  {
	  missing_vals <- nrow(data_subs_2[[list]]) - colSums(data_subs_2[[list]])
      }
       else {  	missing_vals <- nrow(data_subs_2[[list]]) - sum(data_subs_2[[list]])  } 
      missingness[list] <- sum(missing_vals)
    }
	missing_C = sum(missingness)/(dim(MVI)[1]*dim(MVI)[2]) 
    return( missing_C )
  }
  if( ncol(data_train_x) <= 50)
  {
  M <- (ncol(data_train_x))  
  }
  else  { M = 50  }  ## Rarely do we come across more than 20 independent data sources
  
  missing_prop_val <- vector(length=M)
  for(k in 1:M)
  {
    missing_prop_val[k] <- missing_prop(data_present,k, low_threshold)
  }
  return(missing_prop_val)
}


### Get the num_blocks at elbow of plot
## Simple trick: Get the line between first and last point and find the point that is most distant from this line
## https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
num_of_blocks <- function(missing_prop_val)
{
  x_values = seq(1,length(missing_prop_val))
  y_values = missing_prop_val
  
  # Max values to create line
  max_x_x <- max(x_values)
  max_x_y <- y_values[which.max(x_values)]
  max_y_y <- max(y_values)
  max_y_x <- x_values[which.max(y_values)]
  max_df <- data.frame(x = c(max_y_x, max_x_x), y = c(max_y_y, max_x_y))
  
  # Creating straight line between the max values
  fit <- lm(max_df$y ~ max_df$x)
  
  # Distance from point to line
  distances <- c()
  for(i in 1:length(x_values)) {
    distances <- c(distances, abs(coef(fit)[2]*x_values[i] - y_values[i] + coef(fit)[1]) / sqrt(coef(fit)[2]^2 + 1^2))
  }
  
  # Max distance point
  x_max_dist <- x_values[which.max(distances)]
  y_max_dist <- y_values[which.max(distances)]
  
  return(c(x_max_dist))
}


brm_ov_subsets <- function(num_blocks, X_in, Y_in,low_threshold = 0.05, n=5)
{
  data_xy <- cbind(X_in, Y_in)
  data_present <- data.frame( matrix(as.integer(!is.na(X_in)),nrow = nrow(X_in), ncol = ncol(X_in))) 
  names(data_present) <- names(X_in) 
  
  columns_in_subsets <- function(data_xy, MVI,num_blocks,low_threshold, n=5 )
  {
    columns <- list()
	if(n>1)
	{
		determine_best_cluster <- function(data_in,clus_number, n=5)
		{
		  # n <- 10
		  kmeans_o <- lapply(1:n,function(x) { set.seed(x); a<-kmeans(data_in,clus_number); return(a) } ) 
		  opt_kmeans <- which.max(lapply(kmeans_o,function(x) {100-x$tot.withinss/x$totss*100})) 
		  best_cluster <- kmeans_o[[opt_kmeans]]
		  return(best_cluster)
		}	
		member_clusters <- determine_best_cluster(MVI,num_blocks)
	} else { member_clusters <- kmeans(data_in,clus_number) }
	
    members <- member_clusters$cluster
    mcc <- as.data.frame(member_clusters$centers)
    mcc_rounded <- round(mcc,0)
    
    data_subs_1 <- split(MVI,members)  
    data_subs_out <- list(length(data_subs_1))
    data_values <- split(data_xy, members)
    for(list in 1:length(data_subs_1))
    {
      completeness <- colSums(data_subs_1[[list]])
      inputs_in_subset <- which(completeness>(low_threshold)*nrow(data_subs_1[[list]])) 
      ## inputs_in_subset <- which(completeness>(1-low_threshold)*nrow(data_subs_1[[list]]))
      outcome_index <- which(names(data_values[[list]]) ==  names(Y_in) )
      data_subs_out[[list]] <- data_values[[list]][, c(inputs_in_subset,outcome_index) ]
      columns[[list]] <-names(data_subs_out[[list]])
    }
    return(list(mcc_rounded,columns,data_subs_out ))
  }
  
  n_ov_subsetting <- columns_in_subsets(data_xy,data_present,num_blocks,low_threshold)
  cluster_centers_subsets <- n_ov_subsetting[[1]]
  columns_n_ov_subsets <- n_ov_subsetting[[2]]
  n_ov_subsets <- n_ov_subsetting[[3]]
  ## 
  ### Logic for Set theory determination of overlapping subsets
  
  ov_subsets <- list(length(n_ov_subsets))
  
  for(list1 in 1:length(n_ov_subsets))
  {
    ov_subsets[[list1]] <- n_ov_subsets[[list1]]
    for(list2 in 1:length(n_ov_subsets))
    {
      if( all(columns_n_ov_subsets[[list1]] %in% columns_n_ov_subsets[[list2]]) && (list2 != list1 ) )
      {
        subset_cols <- columns_n_ov_subsets[[list1]]
        ov_subsets[[list1]] <- rbind(ov_subsets[[list1]], n_ov_subsets[[list2]][,subset_cols] )
      }
    }
    ov_subsets[[list1]] <- impute_manual(ov_subsets[[list1]])
    n_ov_subsets[[list1]] <- impute_manual(n_ov_subsets[[list1]])
  }
  return(list(ov_subsets,cluster_centers_subsets,n_ov_subsets, columns_n_ov_subsets ))
}

## BRM execution codes in one place as a function:
brm_predictions <- function(num_blocks,X_in, Y_in, data_test,low_threshold = 0.05, n=5)
{
  brm_model <- brm_ov_subsets(num_blocks,X_in, Y_in, low_threshold, n)
  ov_subsets <- brm_model[[1]]
  cluster_centers_subsets <- brm_model[[2]]
  n_ov_subsets <- brm_model[[3]]
  subsets_columns <- brm_model[[4]]
  reg_m <- lapply(ov_subsets,reg_train)
  tree_m <- lapply(ov_subsets,tree_train)
  pred_reg_outcomes <- pred_reduced(reg_m, data_test,data_train_x, data_train_y,cluster_centers_subsets)
  pred_tree_outcomes <- pred_reduced(tree_m, data_test,data_train_x, data_train_y,cluster_centers_subsets)
  return(list(pred_reg_outcomes,pred_tree_outcomes))
}

brm_predictions_n_ov <- function(num_blocks,X_in, Y_in, data_test,low_threshold = 0.05, n=5)
{
  brm_model <- brm_ov_subsets(num_blocks,X_in, Y_in,low_threshold, n)
  ov_subsets <- brm_model[[1]]
  cluster_centers_subsets <- brm_model[[2]]
  n_ov_subsets <- brm_model[[3]]
  subsets_columns <- brm_model[[4]]
  reg_m <- lapply(n_ov_subsets,reg_train)
  tree_m <- lapply(n_ov_subsets,tree_train)
  pred_reg_outcomes <- pred_reduced(reg_m, data_test,data_train_x, data_train_y,cluster_centers_subsets)
  pred_tree_outcomes <- pred_reduced(tree_m, data_test,data_train_x, data_train_y,cluster_centers_subsets)
  return(list(pred_reg_outcomes,pred_tree_outcomes))
}

pred_reduced <- function(models, data_test,X_in, Y_in,cluster_centers_subsets)
{
  data_train <- cbind(X_in, Y_in)
  ## Check test data, to see which is the best matched model subset
  t_data_present <- data.frame( matrix(as.integer(!is.na(data_test)),nrow = nrow(data_test), ncol = ncol(data_test))) 
  names(t_data_present) <- names(data_test) 
  t_data_present <- t_data_present[,!names(t_data_present) %in% names(Y_in)]
  
  nearest_clus <- apply(t_data_present, 1, function(d)
  {
    temp <- apply(cluster_centers_subsets,1,function(x) dist(rbind(x,d)))
    return(which.min(temp))
  }
  )
  
  data_test_in <- data_test
  data_test_in$nearest_clus <- as.factor(nearest_clus)
  data_test_in$rownum <- 1:nrow(data_test_in)
  data_groups <- split(data_test_in,nearest_clus)
  get_back_row_order <- do.call("rbind",data_groups)
  
  data_igroups <- lapply(data_groups,function(x) impute_test(data_train,x))
  reg_outcomes <- list()
  
  for(i in 1: length(data_igroups))
  {   
    data_in <- data_igroups[[i]]
    X <- data_in[,names(data_in) %in% names(X_in)]
    Y <- data_in[,names(data_in) %in% names(Y_in)]
    datam <- data.table(X,Y)	
	reg_outcomes[[i]] <- pred_fun(models[[i]],datam)
  }
  
  reg_out_a <- unlist(reg_outcomes)
  reg_ord <- reg_out_a[order(get_back_row_order[,"rownum"])]
  return(reg_ord)
}

global_weight_fn <- function(num_blocks,X_in, Y_in, low_threshold = 0.05, n=5)
{
  brm_model <- brm_ov_subsets(num_blocks,X_in, Y_in, low_threshold, n)
  ov_subsets <- brm_model[[1]]
  cluster_centers_subsets <- brm_model[[2]]
  n_ov_subsets <- brm_model[[3]]
  subsets_columns <- brm_model[[4]]
  reg_m <- lapply(ov_subsets,reg_train)
  ret_weights <- get_global_wts(reg_m)
  return(ret_weights)
}

## Get the global coefficient scores 
get_global_wts <- function(models)
{
	  global_reg_par <- function(model)
	  {
		nrow <- 
		coef <- summary(model)$coefficients[,1]
		se <- summary(model)$coefficients[,2]
		coef*nrow/se
	  }
	  global_reg_par <- function(model)
		{
	return(model$coefficients*sqrt((length(model$residuals))*length(model$coefficients))/(mse_reg(model)*wt))
		}

	global_reg_par_wt <- function(model)
	{
	return(sqrt((length(model$residuals))*length(model$coefficients))/mse_reg(model))
	}

	wt <- do.call(sum,lapply(models,global_reg_par_wt))

	model_wt_parameters <- lapply(models,global_reg_par)

	df_ <- lapply(model_wt_parameters, function(x) data.frame(t(x)))
	dat_frame <- do.call(smartbind,df_)

    global_wts = colSums(dat_frame,na.rm=T)
	  return(global_wts)  
}

## Get the global coefficient scores 
get_global_wts_gbm <- function(models)
{
  global_gbm_par_wt <- function(model){
    return(sqrt(length(model$var.names)*length(model$fit))/mse_gbm(model))
  }
  
  wt2 <- do.call(sum,lapply(models,global_gbm_par_wt))
  
  global_gbm_par <- function(model){
    varimp <- summary(model)*sqrt(length(model$var.names)*length(model$fit))/(mse_gbm(model)*wt2)
    # variable = data.frame(summary(model)[,1]
    return(varimp)
  }  
  
  gbm_wt_parameters <- lapply(models,global_gbm_par)
  
  weighted_gbm <- do.call(smartbind,lapply(gbm_wt_parameters, function(x) t(x)))
  weighted_gbm_sum <- apply(weighted_gbm,2, function(x) sum(x,na.rm=T))
  
  return(weighted_gbm_sum)  
}
