#' Gradient of the Objective Function Evaluation
#'
#' \code{gradient.g} computes the gradient of the objective function for a given set of parameters.
#'
#' @title Gradient of the Objective Function Evaluation Function
#'
#' @description
#' This function computes the gradient of the objective function based on the provided parameters, which include
#' the source vector (\code{p}), the data matrix (\code{X}), the observed values (\code{y}), the alpha and beta values,
#' and the profile vector (\code{pf.vec}). The gradient is computed based on the specified loss function type.
#'
#' @param p A numeric vector representing the number of variables per source. This defines the structure of the model.
#' @param X A numeric matrix representing the data, where rows correspond to samples and columns represent features.
#' @param y A numeric vector representing the observed values corresponding to the samples.
#' @param alpha A numeric vector or list of alpha values to be optimized during the learning process.
#' @param beta A numeric vector representing the beta coefficients in the model.
#' @param pf.vec A vector indicating the profile or grouping for each sample.
#' @param loss.function A character string specifying the loss function type to use: "ls" for least-square, "log" for logistic regression, or "ce" for cross-entropy.
#'
#' @return
#' The gradient of the objective function, which can be a vector or matrix depending on the loss function.
#' This is computed with respect to the model parameters (\code{alpha} and \code{beta}).
#'
#' @details
#' The function computes the gradient of the objective function based on the loss function type specified by the user.
#' It uses different gradient computation methods for the following loss functions:
#' - **"ls"**: Least-squares loss function, typically used for regression tasks.
#' - **"log"**: Logistic loss function, typically used for binary classification tasks.
#' - **"ce"**: Cross-entropy loss function, used for multi-class classification problems.
#'
#' The gradient is computed with respect to the model parameters (\code{alpha} and \code{beta}), and the result
#' is used in optimization algorithms to update the model parameters during training.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)  # Source vector (number of variables per source)
#' X <- matrix(rnorm(50), nrow = 10, ncol = 5)  # Data matrix
#' y <- c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0)  # Observed values
#' alpha <- c(0.1, 0.2, 0.3)  # Alpha values
#' beta <- c(0.5, -0.2, 0.3, 0.1, -0.4)  # Beta values
#' pf.vec <- c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5)  # Profile vector
#' loss.function <- "log"  # Loss function type ("log" for logistic regression)
#' grad <- gradient.g(p, X, y, alpha, beta, pf.vec, loss.function)
#' print(grad)
#'
#' @seealso
#' \code{\link{gradient.g.ls}} for the gradient computation using least-squares.
#' \code{\link{gradient.g.log}} for the gradient computation using logistic regression.
#' \code{\link{gradient.g.ce}} for the gradient computation using cross-entropy.
#'
#' @export
gradient.g = function(p, X, y, alpha, beta, pf.vec, loss.function){
  switch(
    loss.function,

    # Least-square loss function
    "ls" = return(gradient.g.ls(p, X, y, alpha, beta, pf.vec)),

    # Logistic loss function
    "log" = return(gradient.g.log(p, X, y, alpha, beta, pf.vec)),

    # Cross-entropy loss function
    "ce" = return(gradient.g.ce(p, X, y, alpha, beta, pf.vec))
  )

  return(0)
}
