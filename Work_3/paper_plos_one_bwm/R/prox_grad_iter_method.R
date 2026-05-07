#' Proximal Gradient Iteration Method
#'
#' \code{prox.grad.iter.method} is a function that iteratively computes the beta vector using the proximal gradient method.
#'
#' @title Proximal Gradient Iteration Method
#'
#' @description
#' This function performs proximal gradient iterations to compute the beta vector based on the given parameters and data.
#'
#' @param p The source vector.
#' @param X The data matrix.
#' @param y The target vector.
#' @param alpha The alpha vector.
#' @param beta0 The initial beta vector.
#' @param pf.vec The profile vector.
#' @param lambda The lambda regularization parameter.
#' @param omega The omega value (either "LR" or "RR").
#' @param L.step The L-step factor.
#' @param maxIter The maximum number of iterations.
#' @param tol The tolerance level for convergence.
#' @param gamma The gamma value.
#' @param loss.function The loss function ("ls" or "log").
#'
#' @return
#' The computed beta vector.
#'
#' @details
#' The proximal gradient method is used to compute the beta vector iteratively. This function updates the beta vector until convergence or reaching the maximum number of iterations.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)
#' X <- matrix(runif(100), ncol = 10)
#' y <- rnorm(10)
#' alpha <- c(0.1, 0.2)
#' beta0 <- rep(0, 10)
#' pf.vec <- factor(rep(1:5, each = 2))
#' lambda <- 0.5
#' omega <- "LR"
#' L.step <- 1.5
#' maxIter <- 100
#' tol <- 1e-6
#' gamma <- 0.01
#' loss.function <- "ls"
#' computed_beta <- prox.grad.iter.method(p, X, y, alpha, beta0, pf.vec, lambda, omega, L.step, maxIter, tol, gamma, loss.function)
#'
#' @export
prox.grad.iter.method = function(p, X, y, alpha, beta0, pf.vec,
                                 lambda, omega, L.step, maxIter,
                                 tol, gamma, loss.function){
  tryCatch({
    if(loss.function == 'ce')
      return(prox.grad.iter.method.ce(p, X, y, alpha, beta0, pf.vec,
                                      lambda, omega, L.step, maxIter,
                                      tol, gamma))

    # Function g and its gradient depending just on beta
    func.g <- function(beta){g(p, X, y, alpha, beta, pf.vec, loss.function)}
    grad.g <- function(beta){gradient.g(p, X, y, alpha, beta, pf.vec, loss.function)}

    # Next beta vector
    # We start with L.min = 1
    beta <- beta.suitable.L(beta0, lambda, func.g, grad.g, 1,
                            omega, "reg", L.step, maxIter, tol,
                            p, gamma)
    # Number of iterations
    iter <- 0
    # Repeat until getting solution or achieving maxIter index
    diff.func.beta <- abs(func.g(beta) - func.g(beta0))
    if(is.na(diff.func.beta))
      diff.func.beta <- tol + 1

    Lmin <- 0
    while(diff.func.beta > tol && iter < maxIter){
      # L.min value
      Lmin.aux <- L.min(beta, beta0, grad.g)
      if(is.na(Lmin.aux))
        Lmin.aux <- 1
      if(Lmin.aux > Lmin) Lmin <- Lmin.aux
      # Next beta vector
      beta0 <- beta
      beta <- beta.suitable.L(beta0, lambda, func.g, grad.g, Lmin,
                              omega, "reg", L.step, maxIter, tol,
                              p, gamma)
      if(any(is.na(beta))){
        beta <- beta0
        break
      }
      # Next difference function value and iteration
      diff.func.beta <- abs(func.g(beta) - func.g(beta0))
      if(is.na(diff.func.beta))
        diff.func.beta <- tol + 1
      iter <- iter + 1
    }

    return(beta)
  }, error = function(e) {
    return(beta0)
  })

}
