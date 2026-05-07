#' Proximal Operator for L1 Regularization
#'
#' \code{prox.operator.l1} is a function that computes the proximal operator for L1 regularization.
#'
#' @title Proximal Operator for L1 Regularization
#'
#' @description
#' This function computes the proximal operator for L1 regularization based on the given parameters.
#'
#' @param u The input vector.
#' @param mu The regularization parameter.
#'
#' @return
#' The computed proximal operator for L1 regularization.
#'
#' @details
#' The proximal operator for L1 regularization (also known as soft-thresholding) is used to perform shrinkage on the elements of the input vector u. It applies the element-wise soft-thresholding operation to u with a threshold of mu.
#'
#' @examples
#' # Example usage:
#' u <- c(0.5, -0.2, 0.7)
#' mu <- 0.1
#' prox <- prox.operator.l1(u, mu)
#'
#' @export
prox.operator.l1 = function(u, mu){
  len_u <- length(u)
  # Optimal solution beta
  beta <- numeric(length = len_u)

  # Since the problem is separable, we compute
  # the optimal solution for each component
  for(j in 1:len_u)
    beta[j] <- sign(u[j])*max(abs(u[j]) - mu, 0)

  return(beta)
}
