#' Proximal Operator for L2 Regularization
#'
#' \code{prox.operator.l2} is a function that computes the proximal operator for L2 (ridge) regularization.
#'
#' @title Proximal Operator for L2 Regularization
#'
#' @description
#' This function computes the proximal operator for L2 (ridge) regularization based on the given parameters.
#'
#' @param u The input vector.
#' @param mu The regularization parameter for L2 regularization.
#'
#' @return
#' The computed proximal operator for L2 (ridge) regularization.
#'
#' @details
#' The proximal operator for L2 regularization, also known as ridge regularization, applies a shrinkage operation to the input vector u by dividing it by (1 + mu).
#'
#' @examples
#' # Example usage:
#' u <- c(0.5, -0.2, 0.7, 1.0, -1.5, 2.0, -0.3, 0.8)
#' mu <- 0.1
#' prox <- prox.operator.l2(u, mu)
#'
#' @export
prox.operator.l2 <- function(u, mu){
  # Optimal solution beta
  return(u/(1 + mu))
}
