#' Gradient of the Objective Function Evaluation Function
#'
#' \code{gradient.f} is a function that computes the gradient of the objective function for a given set of parameters.
#'
#' @title Gradient of the Objective Function Evaluation Function
#'
#' @description
#' Computes the gradient of the objective function for a given set of parameters, including observed values (y.m),
#' alpha values (alpha.m), tilde beta values (tilde.beta), and loss function type.
#'
#' @param y.m Observed values.
#' @param alpha.m Alpha values.
#' @param tilde.beta Tilde beta values.
#' @param loss.function Type of loss function ("ls" for least-square, "log" for logistic).
#'
#' @return
#' The gradient of the objective function evaluated with the given parameters.
#'
#' @details
#' This function computes the gradient of the objective function for a given set of parameters and loss function type.
#'
#' @export
gradient.f = function(y.m, alpha.m, tilde.beta, loss.function){
  switch(
    loss.function,

    # Least-square loss function
    "ls" = return(gradient.f.ls(y.m, alpha.m, tilde.beta)),

    # Logistic loss function
    "log" = return(gradient.f.log(y.m, alpha.m, tilde.beta)),
  )

  return(0)
}
