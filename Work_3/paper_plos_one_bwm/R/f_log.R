#' Logistic Loss Function Evaluation Function
#'
#' \code{f.log} is a function that evaluates the logistic loss function for a given set of parameters.
#'
#' @title Logistic Loss Function Evaluation Function
#'
#' @description
#' Evaluates the logistic loss function for a given set of parameters, including observed values (y.m),
#' alpha values (alpha.m), and tilde beta values (tilde.beta).
#'
#' @param y.m Observed values.
#' @param alpha.m Alpha values.
#' @param tilde.beta Tilde beta values.
#'
#' @return
#' The value of the logistic loss function evaluated with the given parameters.
#'
#' @details
#' This function computes the value of the logistic loss function for a given set of parameters.
#'
#' @export
f.log = function(y.m, alpha.m, tilde.beta){
  # Number of sources
  S <- dim(tilde.beta)[2]

  # Value to compute inside norm
  val <- 0
  for(l in 1:dim(tilde.beta)[1])
    val <- val + log(1 + exp(inner.norm.log(y.m, alpha.m, tilde.beta, l)))

  return(val)
}
