#' Gradient of the Logistic Objective Function Evaluation Function
#'
#' \code{gradient.f.log} is a function that computes the gradient of the logistic objective function for a given set of parameters.
#'
#' @title Gradient of the Logistic Objective Function Evaluation Function
#'
#' @description
#' Computes the gradient of the logistic objective function for a given set of parameters, including observed values (y.m),
#' alpha values (alpha.m), and tilde beta values (tilde.beta).
#'
#' @param y.m Observed values.
#' @param alpha.m Alpha values.
#' @param tilde.beta Tilde beta values.
#'
#' @return
#' The gradient of the logistic objective function evaluated with the given parameters.
#'
#' @details
#' This function computes the gradient of the logistic objective function for a given set of parameters. The gradient is
#' computed with respect to the alpha values.
#'
#' @export
gradient.f.log = function(y.m, alpha.m, tilde.beta){
  # Number of sources
  S <- dim(tilde.beta)[2]

  # Gradient of f
  gradient.alpha <- numeric(length = S)
  for(i in 1:S)
    gradient.alpha[i] <- sum(tilde.beta[,i]%*%y.m)
  for(l in 1:dim(tilde.beta)[1])
    gradient.alpha[i] <- gradient.alpha[i] +
    y.m[l]*tilde.beta[l,i]/(1 + exp(inner.norm.log(y.m, alpha.m, tilde.beta, l)))

  return(gradient.alpha)
}
