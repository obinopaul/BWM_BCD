#' Minimum Lipschitz Value Calculation Function
#'
#' \code{L.min} is a function that calculates the minimum Lipschitz constant for a given set of parameters and gradients.
#'
#' @title Minimum Lipschitz Value Calculation Function
#'
#' @description
#' Calculates the minimum Lipschitz constant for a given set of parameters (\code{beta.current} and \code{beta.prev}) and their gradients (\code{gradient}).
#'
#' @param beta.current The current parameter vector.
#' @param beta.prev The previous parameter vector.
#' @param gradient A function that computes the gradient of a specific objective function.
#'
#' @return
#' The minimum Lipschitz constant evaluated with the given parameters and gradients.
#'
#' @details
#' This function calculates the minimum Lipschitz constant, which is used in optimization algorithms to determine step sizes.
# It requires the current and previous parameter vectors (\code{beta.current} and \code{beta.prev}) and a function (\code{gradient}) that computes gradients for a specific objective function.
#'
#' @export
L.min = function(beta.current, beta.prev, gradient){
  diff.beta <- beta.current - beta.prev
  diff.grad.beta <- gradient(beta.current) - gradient(beta.prev)
  return(as.numeric(diff.beta%*%diff.grad.beta/(diff.beta%*%diff.beta)))
}
