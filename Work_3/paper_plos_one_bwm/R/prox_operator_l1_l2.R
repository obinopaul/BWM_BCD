#' Proximal Operator for L1/L2 Regularization
#'
#' \code{prox.operator.l1.l2} is a function that computes the proximal operator for a combination of L1 and L2 regularization.
#'
#' @title Proximal Operator for L1/L2 Regularization
#'
#' @description
#' This function computes the proximal operator for a combination of L1 and L2 regularization based on the given parameters.
#'
#' @param u The input vector.
#' @param mu The regularization parameter for L1 regularization.
#' @param gamma The regularization parameter for L2 regularization.
#'
#' @return
#' The computed proximal operator for the combined L1/L2 regularization.
#'
#' @details
#' The proximal operator for combined L1/L2 regularization applies the proximal operator for L1 regularization followed by L2 regularization on the input vector u. It promotes sparsity within each group defined by the partition vector p and encourages smaller values overall.
#'
#' @examples
#' # Example usage:
#' u <- c(0.5, -0.2, 0.7, 1.0, -1.5, 2.0, -0.3, 0.8)
#' mu <- 0.1
#' gamma <- 0.5
#' prox <- prox.operator.l1.l2(u, mu, gamma)
#'
#' @export
prox.operator.l1.l2 <- function(u, mu, gamma){
  # Optimal solution beta
  return(prox.operator.l2(prox.operator.l1(u, mu), mu*gamma))
}
