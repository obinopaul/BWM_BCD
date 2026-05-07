#' Projection onto Regularized Norm Ball
#'
#' \code{norm.proj.method} is a function that performs the projection of a parameter vector onto a regularized norm ball.
#'
#' @title Projection onto Regularized Norm Ball
#'
#' @description
#' This function performs the projection of a parameter vector \code{beta} onto a regularized norm ball, where the type of norm ball (L1 or L2) and the regularization strength (\code{lambda}) are determined by the user.
#'
#' @param beta The parameter vector to be projected.
#' @param lambda The regularization strength.
#' @param L The Lipschitz constant.
#' @param gradient The gradient of the objective function.
#' @param omega The type of norm ball to project onto ("LR" for L1 norm, "RR" for L2^2 norm).
#' @param tol Tolerance for numerical comparisons (default is 1e-3).
#'
#' @return
#' The projected parameter vector.
#'
#' @details
#' This function calculates the projection of a parameter vector \code{beta} onto a regularized norm ball. The type of norm ball (L1 or L2) and the regularization strength (\code{lambda}) are specified by the user. It uses the Lipschitz constant (\code{L}), gradient, and tolerance (\code{tol}) for the projection.
#'
#' @examples
#' # Example usage:
#' beta <- c(0.5, -0.7, 0.2, 0.1)
#' lambda <- 0.4
#' L <- 0.1
#' gradient <- c(0.1, 0.2, -0.3, 0.4)
#' omega <- "LR"
#' tol <- 1e-3
#' projected_beta <- norm.proj.method(beta, lambda, L, gradient, omega, tol)
#'
#' @export
norm.proj.method = function(beta, lambda, L, gradient, omega, tol = 1e-3){
  # Vector hat beta
  u <- beta - gradient/L

  switch (
    omega,

    # Omega being l1 norm
    "LR" =
      if(sum(abs(u)) > lambda + tol)
        return(prox.operator.l1(u, mu_computation.l1(u, lambda))),

    # Omega being l2^2 norm
    "RR" =
      if(sum(u^2)/2 > lambda + tol)
        return(prox.operator.l2(u, mu_computation.l2(u, lambda))),
  )

  return(u)
}
