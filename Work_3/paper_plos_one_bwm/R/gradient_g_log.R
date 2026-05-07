#' Gradient of the Logistic Objective Function Evaluation Function (Generalized)
#'
#' \code{gradient.g.log} is a function that computes the gradient of the logistic objective function for a given set of parameters in a generalized manner, considering multiple profiles.
#'
#' @title Gradient of the Logistic Objective Function Evaluation Function (Generalized)
#'
#' @description
#' Computes the gradient of the logistic objective function for a given set of parameters, including source vector (p), data matrix (X),
#' observed values (y), alpha values (alpha), beta values (beta), and profile vector (pf.vec).
#'
#' @param p Source vector specifying the number of variables per source.
#' @param X Data matrix.
#' @param y Observed values.
#' @param alpha Alpha values.
#' @param beta Beta values.
#' @param pf.vec Profile vector containing profiles for each sample.
#'
#' @return
#' The gradient of the logistic objective function evaluated with the given parameters.
#'
#' @details
#' This function computes the gradient of the logistic objective function for a given set of parameters. The gradient is
#' computed with respect to the beta values, considering multiple profiles and alpha values. It generalizes the gradient computation for logistic loss.
#'
#' @export
gradient.g.log = function(p, X, y, alpha, beta, pf.vec){
  # Number of sources
  S <- length(p)

  # Profiles
  profiles <- levels(pf.vec)

  # Gradient vector
  grad.vec <- numeric(length = length(beta))
  col.source <- 1
  for(i.source in 1:S){
    # Initialize gradient value
    gradient <- numeric(length = p[i.source])
    next.col.source <- col.source + p[i.source] - 1

    # First value to compute
    for(i in 1:length(profiles)){
      # Profile m
      m <- as.integer(profiles[i])

      # Check if the source is on this profile
      if(!as.binary(m, n = S)[i.source])
        next;

      # Profile m alpha weights
      alpha.m <- alpha[[i]]

      # Samples with current profile
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

      # Values to compute
      alpha.m.i <- alpha.m[i.source]
      X.m.i <- X.m[,col.source:next.col.source]
      val1 <- -t(alpha.m.i*X.m.i)%*%as.matrix(y.m)
      val2 <- numeric(length = p[i.source])
      for(l in 1:dim(X.m)[1])
        val2 <- val2 +
        alpha.m.i*X.m.i[l,]*y.m[l]/(1 + exp(inner.norm.log(y.m, alpha.m, tilde.beta, l)))

      # Gradient update
      gradient <- gradient + (val1 + val2)/dim(X.m)[1]
    }

    grad.vec[col.source:next.col.source] <- gradient
    col.source <- next.col.source + 1
  }

  return(grad.vec/length(profiles))
}
