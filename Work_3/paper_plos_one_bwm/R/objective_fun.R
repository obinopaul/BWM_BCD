#' Objective Function Evaluation
#'
#' \code{objective.fun} is a function that evaluates the objective function for a given model and loss function.
#'
#' @title Objective Function Evaluation
#'
#' @description
#' This function evaluates the objective function for a given model, loss function, and data.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param X The feature matrix.
#' @param y The target vector.
#' @param beta The parameter vector.
#' @param alpha A list of alpha weights.
#' @param pf.vec The profile vector.
#' @param loss.function The loss function ("ls" for least-square or "log" for logistic or "ce" for cross-entropy).
#'
#' @return
#' The value of the objective function.
#'
#' @details
#' This function calculates the value of the objective function based on the given model parameters, data, and loss function. The objective function is used to measure the goodness-of-fit of the model to the data.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)
#' X <- matrix(rnorm(100), ncol = sum(p))
#' y <- rnorm(100)
#' beta <- c(0.5, -0.7, 0.2, 0.1, -0.3)
#' alpha <- list(c(0.1, 0.2), c(0.3, 0.4, 0.5))
#' pf.vec <- as.factor(c(1, 2, 1, 3, 2))
#' loss.function <- "ls"
#' objective <- objective.fun(p, X, y, beta, alpha, pf.vec, loss.function)
#'
#' @export
objective.fun = function(p, X, y, beta, alpha, pf.vec, loss.function){
  switch(
    loss.function,

    # Least-square loss function
    "ls" = return(objective.fun.ls(p, X, y, beta, alpha, pf.vec)),

    # Logistic loss function
    "log" = return(objective.fun.log(p, X, y, beta, alpha, pf.vec)),

    # Entropy loss function
    "ce" = return(objective.fun.ce(p, X, y, beta, alpha, pf.vec)),
  )

  return(0)
}
