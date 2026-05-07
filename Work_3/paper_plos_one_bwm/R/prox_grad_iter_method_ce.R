#' Proximal Gradient Iteration Method for Cross-Entropy Loss
#'
#' \code{prox.grad.iter.method.ce} is a function that iteratively computes the beta vector using the proximal gradient method.
#'
#' @title Proximal Gradient Iteration Method
#'
#' @description
#' This function performs proximal gradient iterations to compute the beta vector for cross-entropy loss in multi-class classification problems.
#'
#' @param p A vector of integers representing the number of features for each source.
#' @param X The data matrix.
#' @param y The target vector of class labels.
#' @param alpha The alpha vector.
#' @param beta0 The initial beta vector.
#' @param pf.vec The profile vector.
#' @param lambda The lambda regularization parameter.
#' @param omega The omega value (either "LR" or "RR").
#' @param L.step The L-step factor.
#' @param maxIter The maximum number of iterations.
#' @param tol The tolerance level for convergence.
#' @param gamma The gamma value.
#'
#' @return
#' The computed beta vector.
#'
#' @details
#' This function implements the proximal gradient method for multi-class classification problems using cross-entropy loss.
#' It iteratively updates the beta vector for each class until convergence or reaching the maximum number of iterations.
#'
#' The function uses helper functions `g`, `gradient.g`, and `beta.suitable.L` which should be defined elsewhere in the code.
#'
#' @note
#' This function is specifically designed for cross-entropy loss and multi-class classification.
#' It assumes that the necessary helper functions are available in the environment.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)
#' X <- matrix(runif(100), ncol = 10)
#' y <- factor(sample(1:3, 10, replace = TRUE))
#' alpha <- c(0.1, 0.2)
#' beta0 <- rep(0, 15)  # 5 features * 3 classes
#' pf.vec <- factor(rep(1:5, each = 2))
#' lambda <- 0.5
#' omega <- "LR"
#' L.step <- 1.5
#' maxIter <- 100
#' tol <- 1e-6
#' gamma <- 0.01
#' computed_beta <- prox.grad.iter.method.ce(p, X, y, alpha, beta0, pf.vec, lambda, omega, L.step, maxIter, tol, gamma)
#'
#' @seealso
#' \code{\link{prox.grad.iter.method}} for the general proximal gradient method.
#'
#' @export
prox.grad.iter.method.ce = function(p, X, y, alpha, beta0, pf.vec,
                                    lambda, omega, L.step, maxIter,
                                    tol, gamma){
  # Function g and its gradient depending just on beta
  func.g <- function(beta){g(p, X, y, alpha, beta, pf.vec, 'ce')}
  grad.g <- function(beta){gradient.g(p, X, y, alpha, beta, pf.vec, 'ce')}

  S <- length(p)
  K <- length(levels(as.factor(y)))
  for(k in K){
    range_beta <- c()
    betaCol <- 1  # Start column index for beta
    for (j in 1:S) {
      nextBetaCol <- betaCol + p[j] * K - 1  # End column for current block in beta
      range_beta <- c(range_beta, (betaCol + p[j] * (k - 1)):(betaCol + p[j] * k - 1))
      betaCol <- nextBetaCol + 1
    }

    # Next beta vector
    # We start with L.min = 1
    beta <- beta.suitable.L(beta0, lambda, func.g, grad.g, 1,
                            omega, "reg", L.step, maxIter, tol,
                            p, gamma, range_beta)
    # Number of iterations
    iter <- 0
    # Repeat until getting solution or achieving maxIter index
    diff.func.beta <- abs(func.g(beta) - func.g(beta0))
    if(is.na(diff.func.beta))
      diff.func.beta <- tol + 1

    Lmin <- 0
    while(diff.func.beta > tol && iter < maxIter){
      # L.min value
      Lmin.aux <- L.min(beta, beta0, grad.g)
      if(is.na(Lmin.aux))
        Lmin.aux <- 1
      if(Lmin.aux > Lmin) Lmin <- Lmin.aux
      # Next beta vector
      beta0 <- beta
      beta <- beta.suitable.L(beta0, lambda, func.g, grad.g, Lmin,
                              omega, "reg", L.step, maxIter, tol,
                              p, gamma, range_beta)
      if(any(is.na(beta))){
        beta <- beta0
        break
      }
      # Next difference function value and iteration
      diff.func.beta <- abs(func.g(beta) - func.g(beta0))
      if(is.na(diff.func.beta))
        diff.func.beta <- tol + 1
      iter <- iter + 1
    }
  }

  return(beta)
}
