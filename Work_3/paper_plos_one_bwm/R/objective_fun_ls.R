#' Objective Function Evaluation for Least-Square Loss
#'
#' \code{objective.fun.ls} is a function that evaluates the objective function for a least-square loss model.
#'
#' @title Objective Function Evaluation for Least-Square Loss
#'
#' @description
#' This function evaluates the objective function for a least-square loss model based on the given model parameters, data, and profile vector.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param X The feature matrix.
#' @param y The target vector.
#' @param beta The parameter vector.
#' @param alpha A list of alpha weights.
#' @param pf.vec The profile vector.
#'
#' @return
#' The value of the least-square loss objective function.
#'
#' @details
#' This function calculates the value of the objective function for a least-square loss model. The objective function is used to measure the goodness-of-fit of the model to the data.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)
#' X <- matrix(rnorm(100), ncol = sum(p))
#' y <- c(0, 1, 0, 1, 0)
#' beta <- c(0.5, -0.7, 0.2, 0.1, -0.3)
#' alpha <- list(c(0.1, 0.2), c(0.3, 0.4, 0.5))
#' pf.vec <- as.factor(c(1, 2, 1, 3, 2))
#' objective <- objective.fun.ls(p, X, y, beta, alpha, pf.vec)
#'
#' @export
objective.fun.ls = function(p, X, y, beta, alpha, pf.vec){
  # Number of sources
  S <- length(p)
  # Profiles
  profiles <- levels(pf.vec)

  # Objective function computing
  obj.func <- 0
  for(i in 1:length(profiles)){
    # Profile m
    m <- as.integer(profiles[i])

    # Profile alpha vec
    alpha.m <- alpha[[i]]

    # Block samples for the profile m
    block.samples <- getBlockSamples(pf.vec, m, S)
    X.m <- X[block.samples$samples,]
    # We update the value inside the norm
    col <- 1
    vec.sum <- numeric(length = dim(X.m)[1])
    for(j in 1:S) {
      nextcol <- col + p[j] - 1
      if(j %in% block.samples$sources)
        vec.sum <- vec.sum + alpha.m[j]*X.m[, col:nextcol]%*%beta[col:nextcol]
      col <- nextcol + 1
    }
    vec.sum <- as.vector(vec.sum) - y[block.samples$samples]

    # We update the value of the objective function
    obj.func <- obj.func + sum(vec.sum^2)/(2*dim(X.m)[1])
  }

  return(obj.func/length(profiles))
}
