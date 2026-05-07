#' Proximal Gradient Method
#'
#' \code{prox.grad.method} is a function that computes the proximal operator of a regularization term.
#'
#' @title Proximal Gradient Method
#'
#' @description
#' This function computes the proximal operator of a regularization term based on the given parameters.
#'
#' @param beta The input vector.
#' @param lambda The regularization parameter.
#' @param L The Lipschitz constant.
#' @param gradient The gradient function.
#' @param omega The regularization type ("LR", "RR", "EN", or "GL").
#' @param p The source vector (optional, required for "GL").
#' @param gamma The gamma value (optional, required for "EN" and "GL").
#'
#' @return
#' The computed proximal operator.
#'
#' @details
#' The proximal gradient method is used to compute the proximal operator of a regularization term. The function returns the proximal operator based on the specified regularization type.
#'
#' @examples
#' # Example usage:
#' beta <- c(0.5, -0.2, 0.7)
#' lambda <- 0.1
#' L <- 1.0
#' gradient <- function(x) 2 * x  # Example gradient function
#' omega <- "LR"
#' computed_prox <- prox.grad.method(beta, lambda, L, gradient, omega)
#'
#' @export
prox.grad.method = function(beta, lambda, L, gradient, omega,
                             p, gamma){
  # Vector hat beta and mu
  u <- beta - gradient/L
  mu <- lambda/L

  switch (
    omega,

    # Omega being l1 norm
    "LR" = return(prox.operator.l1(u, mu)),

    # Omega being l2 norm
    "RR" = return(prox.operator.l2(u, mu)),

    # Omega being l1 + l2^2 norm
    "EN" = return(prox.operator.l1.l2(u, mu, gamma)),

    # Omega being l1/l2 norm
    "GL" = if(!is.null(p)) return(prox.operator.l1_l2(p, u, mu))
    else return(u)
  )

  return(u)
}
