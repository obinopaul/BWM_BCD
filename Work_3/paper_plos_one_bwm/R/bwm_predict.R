#' Predict Function for Block-Wise Missing Data Model
#'
#' \code{bwm.predict} is a function that makes predictions using a block-wise missing data model.
#'
#' @title Predict Function for Block-Wise Missing Data Model
#'
#' @description
#' Makes predictions using a block-wise missing data model based on the provided model and data.
#'
#' @param model A trained block-wise missing data model.
#' @param X Data matrix for prediction.
#' @param p Number of sources.
#'
#' @return
#' A vector of predicted outcomes.
#'
#' @details
#' This function takes a trained block-wise missing data model, input data matrix \code{X}, and the
#' number of sources \code{p} to make predictions. It considers the profile vector of \code{X}
#' and computes predictions based on the model's parameters, including alpha, beta, and profiles.
#'
#' @seealso
#' \code{\link{get_profile}}, \code{\link{as.binary}} for details on profile extraction.
#'
#' @export
bwm.predict = function(model, X, p, verbose = FALSE){
  # Check dimensionality of source vector and matrix X
  if(sum(p) != dim(X)[2]){
    print("The number of variables of the data X do not concide with the provided source vector p.")
    return(NULL)
  }

  # Load libraries
  suppressWarnings(setup(verbose = verbose))

  # Features as matrix
  X <- as.matrix(X)
  if(model$to.normalize)
    for(j in 1:dim(X)[2])
      X[, j] <- (X[, j] - model$translation[j])/model$scale[j]

  # Samples and sources
  n <- dim(X)[1]
  S <- length(p)

  # Predicted outcome
  y.pred <- numeric(length = n)

  # Profiles of data to predict
  pf.vec.pred <- get_profile(p, X)
  pf.vec.pred_num <- as.numeric(levels(pf.vec.pred))[pf.vec.pred]

  K <- length(model$levels.y)
  if(K == 0){
    for(i in 1:n){
      # Profile m of sample i
      m <- pf.vec.pred_num[i]

      # Block sample for profile
      model.profile.index <- which(levels(model$profile.vector) == m)
      if(length(model.profile.index) == 0)
        y.pred[i] <- NA
      else {
        sources.profile <- which(as.binary(m, n = S))
        model.profile.index <- as.integer(model.profile.index[1])
        col <- 1
        for(j in 1:S){
          nextCol <- col + p[j] - 1
          if(j %in% sources.profile)
            y.pred[i] <- y.pred[i] + model$alpha[[model.profile.index]][j]*
              X[i, col:nextCol]%*%model$beta[col:nextCol]
          col <- nextCol + 1
        }
      }
    }
  } else {
    if(K == 1){
      return(rep(1, n))
    }else if(K == 2){
      levels <- c(-1,1)
      for(i in 1:n){
        # Profile m of sample i
        m <- pf.vec.pred_num[i]

        # Block sample for profile
        model.profile.index <- which(levels(model$profile.vector) == m)
        if(length(model.profile.index) == 0)
          y.pred[i] <- NA
        else {
          sources.profile <- which(as.binary(m, n = S))
          model.profile.index <- as.integer(model.profile.index[1])

          # Model value
          col <- 1
          model.value <- 0
          for(j in 1:S){
            nextCol <- col + p[j] - 1
            if(j %in% sources.profile)
              model.value <- model.value + model$alpha[[model.profile.index]][j]*
                X[i, col:nextCol]%*%model$beta[col:nextCol]
            col <- nextCol + 1
          }

          # Probabilities
          optim.values <- numeric(length = 2)
          optim.values[1] <- log(1 + exp(-model.value*levels[1]))
          optim.values[2] <- log(1 + exp(-model.value*levels[2]))

          y.pred[i] <- model$levels.y[which.min(optim.values)]
        }
      }
    }
    else{
      levels <- 1:K
      beta <- numeric()
      colBeta <- 1
      for(i in 1:S)
        for(k in 1:K){
          nextColBeta <- colBeta + p[i] - 1
          n_range <- colBeta:nextColBeta
          beta[n_range] <- model$beta[[i]][,k]
          colBeta <- nextColBeta + 1
        }

      for(i in 1:n){
        # Profile m of sample i
        m <- pf.vec.pred_num[i]
        # Block sample for profile
        model.profile.index <- which(levels(model$profile.vector) == m)
        if(length(model.profile.index) == 0)
          y.pred[i] <- NA
        else {
          block.samples <- getBlockSamples(pf.vec.pred, m, S)
          optim.values <- numeric(length = K)
          numerator <- 0
          alpha.m <- model$alpha[[model.profile.index]]
          row.X <- X[i, ]
          for(k in 1:K)
            optim.values[k] <- exp(sigma_function(p, alpha.m, row.X,
                                                  k, beta, K, block.samples))

          # Normalize the probabilities by dividing by their sum
          optim.values <- optim.values / sum(optim.values)

          # Compute the log-probabilities
          log_optim.values <- -log(optim.values)

          # Save result
          y.pred[i] <- model$levels.y[which.min(log_optim.values)]
        }
      }
    }
  }

  return(y.pred)
}
