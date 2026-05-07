#' Gradient of the Least-Square Objective Function Evaluation Function (Generalized)
#'
#' \code{gradient.g.ls} is a function that computes the gradient of the least-square objective function for a given set of parameters in a generalized manner, considering multiple profiles.
#'
#' @title Gradient of the Least-Square Objective Function Evaluation Function (Generalized)
#'
#' @description
#' Computes the gradient of the least-square objective function for a given set of parameters, including source vector (p), data matrix (X),
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
#' The gradient of the least-square objective function evaluated with the given parameters.
#'
#' @details
#' This function computes the gradient of the least-square objective function for a given set of parameters. The gradient is
#' computed with respect to the beta values, considering multiple profiles and alpha values. It generalizes the gradient computation for least-square loss.
#'
#' @export
gradient.g.ls = function(p, X, y, alpha, beta, pf.vec){
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

      # First value to compute
      val1 <- numeric(length = dim(X.m)[1])
      col <- 1
      for(j in 1:S){
        nextcol <- col + p[j] - 1
        if(j %in% block.samples$sources)
          val1 <- val1 + alpha.m[j]*(X.m[, col:nextcol]%*%beta[col:nextcol])
        col <- nextcol + 1
      }
      val1 <- val1 - y[block.samples$samples]

      # Second value to compute
      val2 <- t(alpha.m[i.source]*X.m[,col.source:next.col.source])

      # Gradient update
      gradient <- gradient + (val2%*%val1)/dim(X.m)[1]
    }

    grad.vec[col.source:next.col.source] <- gradient
    col.source <- next.col.source + 1
  }

  return(grad.vec/length(profiles))
}
