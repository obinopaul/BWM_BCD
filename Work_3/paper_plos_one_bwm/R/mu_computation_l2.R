#' Compute Regularization Parameter mu for L2 Regularization
#'
#' \code{mu_computation.l2} is a function that computes the regularization parameter \eqn{\mu} for L2 regularization.
#'
#' @title Compute Regularization Parameter mu for L2 Regularization
#'
#' @description
#' Computes the regularization parameter \eqn{\mu} for L2 regularization based on the current parameter vector (\code{beta}) and the regularization strength (\code{lambda}).
#'
#' @param beta The current parameter vector.
#' @param lambda The regularization strength.
#'
#' @return
#' The computed regularization parameter \eqn{\mu} for L2 regularization.
#'
#' @details
#' This function calculates the regularization parameter \eqn{\mu} for L2 regularization based on the input parameter vector \code{beta} and the regularization strength \code{lambda}.
# It uses a formula to compute \eqn{\mu} from the L2-norm of the parameter vector.
#'
#' @examples
#' # Example usage:
#' beta <- c(0.5, -0.7, 0.2, 0.1)
#' lambda <- 0.4
#' mu <- mu_computation.l2(beta, lambda)
#'
#' @export
mu_computation.l2 = function(beta, lambda){
  return(sqrt(sum(beta^2)/(2*lambda)) - 1)
}
