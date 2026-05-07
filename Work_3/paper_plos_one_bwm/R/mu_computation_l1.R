#' Compute Regularization Parameter mu for L1 Regularization
#'
#' \code{mu_computation.l1} is a function that computes the regularization parameter \eqn{\mu} for L1 regularization.
#'
#' @title Compute Regularization Parameter mu for L1 Regularization
#'
#' @description
#' Computes the regularization parameter \eqn{\mu} for L1 regularization based on the current parameter vector (\code{beta}) and the regularization strength (\code{lambda}).
#'
#' @param beta The current parameter vector.
#' @param lambda The regularization strength.
#'
#' @return
#' The computed regularization parameter \eqn{\mu} for L1 regularization.
#'
#' @details
#' This function calculates the regularization parameter \eqn{\mu} for L1 regularization based on the input parameter vector \code{beta} and the regularization strength \code{lambda}.
# It iteratively seeks for the index \code{k} and computes \eqn{\mu} accordingly.
#'
#' @examples
#' # Example usage:
#' beta <- c(0.5, -0.7, 0.2, 0.1)
#' lambda <- 0.4
#' mu <- mu_computation.l1(beta, lambda)
#'
#' @export
mu_computation.l1 = function(beta, lambda){
  # Define vector b
  b <- c(abs(beta), 0)
  b <- b[order(b, decreasing = TRUE)]

  # Seeking for the index k
  k <- 2
  S.bk <- sum(abs(prox.operator.l1(beta, b[k])))
  # Do the loop until the index k is found
  while(lambda > S.bk){
    k <- k + 1
    S.bk <- sum(abs(prox.operator.l1(beta, b[k])))
  }

  k <- k - 1
  S.bk <- sum(abs(prox.operator.l1(beta, b[k])))
  return(b[k] - (lambda - S.bk)/k)
}
