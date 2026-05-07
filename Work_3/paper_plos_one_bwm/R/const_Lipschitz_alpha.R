#' Compute Lipschitz Constant for Alpha Optimization
#'
#' \code{const.Lipschitz.alpha} is a function that computes the Lipschitz constant for
#' alpha optimization based on the provided tilde beta values.
#'
#' @title Compute Lipschitz Constant for Alpha Optimization
#'
#' @description
#' Computes the Lipschitz constant for alpha optimization using the provided tilde beta values.
#'
#' @param tilde.beta Matrix of tilde beta values.
#'
#' @return
#' The Lipschitz constant for alpha optimization.
#'
#' @details
#' This function calculates the Lipschitz constant used in the alpha optimization process.
#' It takes a matrix of tilde beta values as input and computes the maximum sum of squares
#' of columns in the matrix as the Lipschitz constant.
#'
#' @export
const.Lipschitz.alpha = function(tilde.beta){
  sum.sq <- numeric(length = dim(tilde.beta)[2])
  for(j in 1:dim(tilde.beta)[2])
    sum.sq[j] <- sum(tilde.beta[,j]^2)

  return(max(sum.sq))
}
