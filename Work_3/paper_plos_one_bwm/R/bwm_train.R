#' Block-Wise Missing Data Model Training Function
#'
#' \code{bwm.train} is a function that trains a block-wise missing data model using the provided data
#'
#' @title Block-Wise Missing Data Model Training Function
#'
#' @description
#' Trains a block-wise missing data model using the provided data and user-defined parameters.
#'
#' @param p Number of sources.
#' @param X Data matrix for training.
#' @param y Response vector for training.
#' @param lambda Vector of regularization parameters.
#' @param L.step Learning step size for L.
#' @param maxIter Maximum number of iterations for training.
#' @param tol Tolerance level for convergence.
#' @param omega.alpha Alpha regularization method ("LR" or other).
#' @param tol.alpha Tolerance level for alpha convergence.
#' @param maxIter.alpha Maximum number of iterations for alpha optimization.
#' @param omega.beta Beta regularization method ("LR" or other).
#' @param alpha0.comp Method for alpha initialization ("equiv" or other).
#' @param tol.beta Tolerance level for beta convergence.
#' @param maxIter.beta Maximum number of iterations for beta optimization.
#' @param gamma Tuning parameter for optimization.
#' @param to.normalize Logical indicating whether to normalize the features.
#' @param beta0 Optional initial beta values (list or vector).
#' @param alpha0 Optional initial alpha values (list or vector).
#' @param maxClasses Optional number of max classes to classify (int).
#'
#' @return
#' A list containing the trained block-wise missing data model parameters.
#'
#' @details
#' This function trains a block-wise missing data model using the provided data and user-defined
#' parameters. It supports various options for regularization, initialization, and optimization.
#'
#' @seealso
#' \code{\link{alpha.initialization}}, \code{\link{beta.initialization}},
#' \code{\link{prox.grad.iter.method}} for details on initialization and optimization methods.
#'
#' @export
bwm.train = function(p, X, y, lambda, L.step = 1.5, maxIter = 300, tol = 1e-12,
                     omega.alpha = "LR", tol.alpha = 1e-12, maxIter.alpha = 20,
                     omega.beta = "LR", alpha0.comp = "equiv",
                     tol.beta = 1e-12, maxIter.beta = 20, gamma = 1, to.normalize = F,
                     beta0, alpha0, verbose = TRUE, maxClasses = 20){

  # Load libraries
  suppressWarnings(setup(verbose = verbose))

  # Check dimensionality of source vector and matrix X
  if(sum(p) != dim(X)[2]){
    print("The number of variables of the data X do not concide with the provided source vector p.")
    return(NULL)
  }

  # Initializes the progress bar
  pb <- txtProgressBar(min = 0, # Minimum value of the progress bar
                       max = maxIter*length(lambda), # Maximum value of the progress bar
                       style = 3,    # Progress bar style
                       width = 50,   # Progress bar width
                       char = "=")   # Character used to create the bar

  # L.step factor definition
  L.step <- max(1.001, L.step)

  # Features
  X <- as.matrix(X)
  translation <- rep(0, dim(X)[2])
  scale <- rep(1, dim(X)[2])
  if(to.normalize){
    for(j in 1:dim(X)[2]){
      x <- X[, j]
      # Check if the column is numeric
      if(!is.numeric(x))
        next

      # Remove NA values for min and max calculation
      x_non_na <- x[!is.na(x)]

      if (length(x_non_na) == 0)
        next  # Skip scaling if the column is all NA

      # Calculate min and max
      min.x <- min(x_non_na)
      max.x <- max(x_non_na)

      # Store translation and scale
      translation[j] <- min.x
      scale[j] <- max.x - min.x

      # Check for zero range
      if (scale[j] == 0)
        scale[j] <- 1  # Avoid division by zero

      # Scale the column, preserving NA values
      X[, j] <- (x - translation[j]) / scale[j]
    }
  }

  # Outcome
  lev.y <- c()
  loss.function <- "ls"
  beta0.comp = "LMR"
  if(length(unique(y)) > 1 && length(unique(y)) < maxClasses){
    aux_y <- as.factor(y)
    lev.y <- levels(aux_y)
    if(length(levels(aux_y)) == 2){
      aux_y <- lev.y[as.numeric(aux_y)]
      aux_y[aux_y == lev.y[1]] <- -1
      aux_y[aux_y == lev.y[2]] <- 1
      y <- as.numeric(aux_y)

      loss.function <- "log"
      beta0.comp <- "LogR"
    }else{
      y <- as.numeric(aux_y)
      loss.function <- "ce"
      beta0.comp <- "MultinomR"
    }
  }

  # Number of sources
  S <- length(p)

  # We compute the profiles
  pf.vec <- get_profile(p, X)

  # If it is complete data, alpha weights are fixed
  keep.alpha <- length(levels(pf.vec)) == 1

  # Best alpha, beta and lambda parameters
  if(missing(alpha0))
    best.alpha <- alpha.initialization(pf.vec, S, keep.alpha, alpha0.comp)
  else if(is.list(alpha0)) best.alpha <- alpha0
  else best.alpha <- as.list(alpha0)
  if(missing(beta0))
    best.beta <- beta.initialization(p, X, y, beta0.comp)
  else if(is.list(beta0)) best.beta <- beta0
  else best.beta <- as.list(beta0)
  best.lambda <- NA

  # Best objective function value
  obj.func.best <- objective.fun(p, X, y, best.beta, best.alpha, pf.vec, loss.function)
  for(j in 1:length(lambda)){
    converge <- FALSE
    # Initial objective function value
    obj.func0 <- obj.func.best
    # We initialize alpha0 weights
    alpha0 <- best.alpha
    # We initialize beta0 models
    beta0 <- best.beta

    # We compute the optimal beta
    alpha <- alpha0
    for(k in 1:maxIter){
      # If alpha is not always the same
      if(!keep.alpha)
        # Computing alpha when beta is fixed
        alpha <- alpha.compute(p, X, y, beta0, alpha0, pf.vec,
                               omega.alpha, L.step, maxIter.alpha,
                               tol.alpha, loss.function)


      # Computing beta when alpha is fixed
      beta <- prox.grad.iter.method(p, X, y, alpha, beta0, pf.vec,
                                    lambda[j], omega.beta, L.step,
                                    maxIter.beta, tol.beta, gamma,
                                    loss.function)

      # Objective function computation
      obj.func <- objective.fun(p, X, y, beta, alpha, pf.vec, loss.function)
      # If the objective stops decreasing, we stop computing
      if(!is.na(obj.func) && abs(obj.func - obj.func0) < tol){
        if(obj.func < obj.func0){
          # We update both alpha and beta vectors
          beta0 <- beta
          alpha0 <- alpha
          # and the objective function value
          obj.func0 <- obj.func
        }

        converge <- TRUE
        break;
      }

      # Otherwise, we update both alpha and beta vectors
      beta0 <- beta
      alpha0 <- alpha
      # and the objective function value
      obj.func0 <- obj.func

      # Sets the progress bar to the current state
      setTxtProgressBar(pb, k + (j - 1)*maxIter)
    }

    # Get best parameters
    if(!is.na(obj.func0))
      if(obj.func0 < obj.func.best){
        best.beta <- beta0
        best.alpha <- alpha0
        best.lambda <- lambda[j]
        obj.func.best <- obj.func0
      }

    # Sets the progress bar to the current state
    setTxtProgressBar(pb, j*maxIter)
    if(converge)
      cat(paste0("\nFor the parameter lambda = ", lambda[j], " the algorithm has converged before reaching the maximum number of iterations."))
    else
      cat(paste0("\nFor the parameter lambda = ", lambda[j], " the algorithm has reached the maximum number of iterations."))
  }

  if(loss.function == 'ce'){
    S <- length(p)
    K <- length(levels(as.factor(y)))
    final.beta <- list()
    col <- 1
    colBeta <- 1
    for(i in 1:S){
      aux.beta <- matrix(0, nrow = p[i], ncol = K)
      aux.colBeta <- colBeta
      for(k in 1:K){
        aux.nextColBeta <- aux.colBeta + p[i] - 1
        n_range <- aux.colBeta:aux.nextColBeta
        aux.beta[,k] <- best.beta[n_range]
        aux.colBeta <- aux.nextColBeta + 1
      }
      colBeta <- colBeta + p[i]*K
      final.beta[[i]] <- aux.beta
    }

    best.beta <- final.beta
  }

  # Ending progress bar
  setTxtProgressBar(pb, length(lambda)*maxIter)
  # Final coefficients
  result <- list(alpha = best.alpha, beta = best.beta,
                 lambda = best.lambda, profile.vector = pf.vec,
                 to.normalize = to.normalize, translation = translation,
                 scale = scale, levels.y = lev.y)
  return(result)
}
