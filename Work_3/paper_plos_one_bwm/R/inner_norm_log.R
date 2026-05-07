#' Inner Normalization Function (Generalized)
#'
#' \code{inner.norm.log} is a function that computes the inner normalization term for a given set of parameters in a generalized manner, considering multiple sources.
#'
#' @title Inner Normalization Function (Generalized)
#'
#' @description
#' Computes the inner normalization term for a given set of parameters, including observed values (y.m), alpha values (alpha.m), and beta values (tilde.beta).
#'
#' @param y.m Observed values.
#' @param alpha.m Alpha values.
#' @param tilde.beta Beta values.
#' @param l Index for the current observation.
#'
#' @return
#' The inner normalization term evaluated with the given parameters.
#'
#' @details
#' This function computes the inner normalization term for a given set of parameters, considering multiple sources and alpha values.
#'
#' @export
inner.norm.log = function(y.m, alpha.m, tilde.beta, l){
  # Number of sources
  S <- dim(tilde.beta)[2]
  val <- 0
  for(i in 1:S)
    val <- val + alpha.m[i]*tilde.beta[l,i]

  return(-val*y.m[l])
}
