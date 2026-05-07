#' Suitable Beta Computation Function
#'
#' \code{beta.suitable.L} is a function that computes a suitable beta vector for a given lambda
#' value and optimization method.
#'
#' @title Suitable Beta Computation Function
#'
#' @description
#' Computes a suitable beta vector for a given lambda value and optimization method by
#' iteratively adjusting the regularization parameter.
#'
#' @param beta Initial beta vector.
#' @param lambda Regularization parameter.
#' @param function.L A function to calculate the objective value.
#' @param gradient.L A function to calculate the gradient vector.
#' @param L.min Minimum regularization parameter (L).
#' @param omega Omega parameter.
#' @param optimization Optimization method ("reg" or "cons").
#' @param L.step Learning step size for adjusting L.
#' @param maxIter Maximum number of iterations.
#' @param tol Tolerance level for convergence.
#' @param p Number of sources (optional).
#' @param gamma Tuning parameter (optional).
#'
#' @return
#' A suitable beta vector for the given lambda value and optimization method.
#'
#' @details
#' This function iteratively computes a suitable beta vector for a given lambda value
#' and optimization method. It adjusts the regularization parameter L and computes the
#' beta vector that minimizes the objective function within the given tolerance.
#'
#' @seealso
#' \code{\link{prox.grad.method}}, \code{\link{norm.proj.method}} for details on the optimization methods.
#'
#' @export
beta.suitable.L = function(beta, lambda, function.L, gradient.L, L.min,
                           omega, optimization, L.step, maxIter, tol,
                           p = NULL, gamma = 1, range_beta = NULL){
  if(is.null(range_beta))
    range_beta <- 1:length(beta)

  # Compute gradient vector evaluated at beta
  gradient <- gradient.L(beta)

  # Compute objective value evaluated at beta
  objective <- function.L(beta)

  # Choose framework
  method.beta.star <- switch(
    optimization,

    "reg" = function(L, beta){return(prox.grad.method(beta, lambda, L,
                                                      gradient[range_beta],
                                                      omega, p, gamma))},

    "cons" = function(L, beta){return(norm.proj.method(beta, lambda, L,
                                                       gradient[range_beta],
                                                       omega))},
  )

  # Compute beta star from L
  L <- L.min
  aux.beta.star <- method.beta.star(L, beta[range_beta])
  beta.star <- beta
  beta.star[range_beta] <- aux.beta.star
  # Linearization of objective
  diff.beta <- beta.star - beta
  linear.L <- as.numeric(objective - function.L(beta.star) + gradient%*%diff.beta + L/2*sum(diff.beta^2))
  if(is.na(linear.L))
    linear.L <- 0
  iter <- 0
  while(iter < maxIter && linear.L < tol){
    # Compute next beta star from L
    L <- L*L.step
    aux.beta.star <- method.beta.star(L, beta[range_beta])
    beta.star <- beta
    beta.star[range_beta] <- aux.beta.star

    # Linearization of objective
    diff.beta <- beta.star - beta
    linear.L <- as.numeric(objective - function.L(beta.star) + gradient%*%diff.beta + L/2*sum(diff.beta^2))
    if(is.na(linear.L))
      linear.L <- 0

    iter <- iter + 1
  }

  return(beta.star)
}
