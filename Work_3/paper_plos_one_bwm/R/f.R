#' Loss Function Evaluation Function
#'
#' \code{f} is a function that evaluates a loss function for a given set of parameters.
#'
#' @title Loss Function Evaluation Function
#'
#' @description
#' Evaluates a loss function for a given set of parameters, including observed values (y.m),
#' alpha values (alpha.m), tilde beta values (tilde.beta), and the loss function type.
#'
#' @param y.m Observed values.
#' @param alpha.m Alpha values.
#' @param tilde.beta Tilde beta values.
#' @param loss.function Type of loss function ("ls" for least-square, "log" for logistic).
#'
#' @return
#' The value of the loss function evaluated with the given parameters.
#'
#' @details
#' This function computes the value of a loss function for a given set of parameters based on
#' the specified loss function type. It supports both least-square (ls) and logistic (log) loss
#' functions.
#'
#' @seealso
#' \code{\link{f.ls}}, \code{\link{f.log}} for details on specific loss functions.
#'
#' @export
f = function(y.m, alpha.m, tilde.beta, loss.function){
  switch(
    loss.function,

    # Least-square loss function
    "ls" = return(f.ls(y.m, alpha.m, tilde.beta)),

    # Logistic loss function
    "log" = return(f.log(y.m, alpha.m, tilde.beta)),
  )

  return(0)
}
