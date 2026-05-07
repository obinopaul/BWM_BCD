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
omega.norm.proj.method_ce = function(p, alpha.m, X.m, y.m, beta, block.samples, K,
                                     omega, L.step, maxIter, tol){
  # Function f and its gradient depending just on alpha
  func.f <- function(alpha){f.ce(p, alpha, X.m, y.m, beta, block.samples, K)}
  grad.f <- function(alpha){gradient.f.ce(p, alpha, X.m, y.m, beta, block.samples, K)}

  # Lipschitz constant computation based on the feature matrix X.m and sample size length(y.m)
  Lmin <- min(c(compute_lipschitz_alpha.ce(X.m, length(y.m)), 1))

  # Next alpha vector
  alpha <- beta.suitable.L(alpha.m, 1, func.f, grad.f, 1, omega, "cons",
                           L.step, maxIter, tol)

  # Number of iterations
  iter <- 0

  # Repeat until getting solution or achieving maxIter index
  diff.func.alpha <- abs(func.f(alpha) - func.f(alpha.m))
  if(is.na(diff.func.alpha))
    return(alpha.m)
  while(diff.func.alpha > tol && iter < maxIter){
    # Update alpha0
    alpha.m <- alpha
    alpha <- beta.suitable.L(alpha.m, 1, func.f, grad.f, Lmin, omega,
                             "cons", L.step, maxIter, tol)

    # Update function difference and iteration count
    diff.func.alpha <- abs(func.f(alpha) - func.f(alpha.m))
    iter <- iter + 1
  }

  return(alpha)
}
