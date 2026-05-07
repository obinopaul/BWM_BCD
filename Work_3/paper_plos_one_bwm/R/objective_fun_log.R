#' Objective Function Evaluation for Logistic Loss
#'
#' \code{objective.fun.log} is a function that evaluates the objective function for a logistic loss model.
#'
#' @title Objective Function Evaluation for Logistic Loss
#'
#' @description
#' This function evaluates the objective function for a logistic loss model based on the given model parameters, data, and profile vector.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param X The feature matrix.
#' @param y The target vector.
#' @param beta The parameter vector.
#' @param alpha A list of alpha weights.
#' @param pf.vec The profile vector.
#'
#' @return
#' The value of the logistic loss objective function.
#'
#' @details
#' This function calculates the value of the objective function for a logistic loss model. The objective function is used to measure the goodness-of-fit of the model to the data.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)
#' X <- matrix(rnorm(100), ncol = sum(p))
#' y <- c(0, 1, 0, 1, 0)
#' beta <- c(0.5, -0.7, 0.2, 0.1, -0.3)
#' alpha <- list(c(0.1, 0.2), c(0.3, 0.4, 0.5))
#' pf.vec <- as.factor(c(1, 2, 1, 3, 2))
#' objective <- objective.fun.log(p, X, y, beta, alpha, pf.vec)
#'
#' @export
objective.fun.log = function(p, X, y, beta, alpha, pf.vec){
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

    # Outcome with current profile
    y.m <- y[block.samples$samples]

    # Prediction matrix from sample
    tilde.beta <- numeric()
    col <- 1
    for(j in 1:S){
      nextCol <- col + p[j] - 1
      if(j %in% block.samples$sources)
        tilde.beta <- cbind(tilde.beta, X.m[, col:nextCol]%*%beta[col:nextCol])
      else tilde.beta <- cbind(tilde.beta, rep(0, dim(X.m)[1]))

      col <- nextCol + 1
    }

    val <- 0
    for(l in 1:dim(tilde.beta)[1])
      val <- val + log(1 + exp(inner.norm.log(y.m, alpha.m, tilde.beta, l)))

    # We update the value of the objective function
    obj.func <- obj.func + val/dim(tilde.beta)[1]
  }

  return(obj.func/length(profiles))
}
