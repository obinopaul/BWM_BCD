#' Least-Square Loss Function Evaluation Function
#'
#' \code{f.ls} is a function that evaluates the least-square loss function for a given set of parameters.
#'
#' @title Least-Square Loss Function Evaluation Function
#'
#' @description
#' Evaluates the least-square loss function for a given set of parameters, including observed values (y.m),
#' alpha values (alpha.m), and tilde beta values (tilde.beta).
#'
#' @param y.m Observed values.
#' @param alpha.m Alpha values.
#' @param tilde.beta Tilde beta values.
#'
#' @return
#' The value of the least-square loss function evaluated with the given parameters.
#'
#' @details
#' This function computes the value of the least-square loss function for a given set of parameters.
#'
#' @export
f.ls = function(y.m, alpha.m, tilde.beta){
  # Number of sources
  S <- dim(tilde.beta)[2]

  # Value to compute inside norm
  val <- numeric(length = length(y.m))
  for(j in 1:S)
    val <- val + alpha.m[j]*tilde.beta[,j]
  val <- val - y.m

  return(sum(val^2)/2)
}
