#' Omega-Norm Projection Method
#'
#' \code{omega.norm.proj.method} is a function that performs the omega-norm projection method to compute the alpha vector.
#'
#' @title Omega-Norm Projection Method
#'
#' @description
#' This function applies the omega-norm projection method to compute the alpha vector based on the given parameters and data.
#'
#' @param y.m The target vector for a specific profile.
#' @param alpha0 The initial alpha vector.
#' @param tilde.beta The parameter matrix.
#' @param omega The omega value (either "LR" or "RR").
#' @param L.step The L-step factor.
#' @param maxIter The maximum number of iterations.
#' @param tol The tolerance level for convergence.
#' @param loss.function The loss function ("ls" or "log").
#'
#' @return
#' The computed alpha vector.
#'
#' @details
#' The omega-norm projection method is used to compute the alpha vector in a model. This function iteratively updates the alpha vector until convergence or reaching the maximum number of iterations.
#'
#' @examples
#' # Example usage:
#' y.m <- c(0, 1, 0, 1, 0)
#' alpha0 <- c(0.1, 0.2)
#' tilde.beta <- matrix(c(0.5, -0.7, 0.2, 0.1, -0.3), ncol = 2)
#' omega <- "LR"
#' L.step <- 1.5
#' maxIter <- 100
#' tol <- 1e-6
#' loss.function <- "log"
#' computed_alpha <- omega.norm.proj.method(y.m, alpha0, tilde.beta, omega, L.step, maxIter, tol, loss.function)
#'
#' @export
omega.norm.proj.method = function(y.m, alpha0, tilde.beta, omega, L.step,
                                   maxIter, tol, loss.function){
  # Function f and its gradient depending just on alpha
  func.f <- function(alpha){f(y.m, alpha, tilde.beta, loss.function)}
  grad.f <- function(alpha){gradient.f(y.m, alpha, tilde.beta, loss.function)}
  # First L.min value
  Lmin <- const.Lipschitz.alpha(tilde.beta)

  # Next alpha vector
  alpha <- beta.suitable.L(alpha0, 1, func.f, grad.f, 1, omega, "cons",
                           L.step, maxIter, tol)
  # Number of iterations
  iter <- 0
  # Repeat until getting solution or achieving maxIter index
  diff.func.alpha <- abs(func.f(alpha) - func.f(alpha0))
  while(diff.func.alpha > tol && iter < maxIter){
    # Next alpha vector
    alpha0 <- alpha
    alpha <- beta.suitable.L(alpha0, 1, func.f, grad.f, Lmin, omega,
                             "cons", L.step, maxIter, tol)

    # Next difference function value and iteration
    diff.func.alpha <- abs(func.f(alpha) - func.f(alpha0))
    iter <- iter + 1
  }

  return(alpha)
}
