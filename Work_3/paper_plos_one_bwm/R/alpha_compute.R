#' Alpha Computation Function
#'
#' \code{alpha.compute} computes the alpha values for each profile using optimization
#' and projection methods tailored for specific loss functions.
#'
#' @title Alpha Computation Function
#'
#' @description
#' This function calculates updated alpha values for each profile in the dataset. Alpha values
#' represent the relative importance of different sources (blocks) within a profile. The function
#' iterates through profiles, selects relevant samples and blocks, and updates alpha values using
#' a projection-based optimization method. The approach adapts to the specified loss function,
#' such as cross-entropy (\code{"ce"}) or a custom loss.
#'
#' @param p A numeric vector specifying the number of variables for each source (block).
#' @param X A numeric matrix of predictors (features).
#' @param y A numeric or factor vector of response variables (e.g., class labels for classification).
#' @param beta A numeric vector of coefficients for all blocks.
#' @param alpha0 A list of initial alpha values for each profile.
#' @param pf.vec A vector assigning each sample to a specific profile.
#' @param omega A numeric value or vector representing the norm constraint for the projection.
#' @param L.step A numeric value for the learning step size.
#' @param maxIter An integer specifying the maximum number of iterations for optimization.
#' @param tol A numeric value for the tolerance level to determine convergence.
#' @param loss.function A string specifying the loss function to optimize (e.g., \code{"ce"} for cross-entropy).
#'
#' @return
#' A list where each element contains the computed alpha values for the corresponding profile.
#'
#' @details
#' The function processes each profile individually:
#' - For cross-entropy loss (\code{"ce"}), the function extracts the samples for the current profile
#'   and computes updated alpha values using the \code{\link{omega.norm.proj.method_ce}} function.
#' - For other loss functions, a prediction matrix is constructed, and the \code{\link{omega.norm.proj.method}}
#'   is used for updating alpha values.
#'
#' The projection method ensures that alpha values respect the specified norm constraint (\code{omega}).
#'
#' @seealso
#' \code{\link{omega.norm.proj.method}} for general projection-based optimization.
#' \code{\link{omega.norm.proj.method_ce}} for cross-entropy-specific projection optimization.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)  # Two sources with 2 and 3 variables respectively
#' X <- matrix(rnorm(50), nrow = 10, ncol = 5)  # Feature matrix
#' y <- sample(1:3, 10, replace = TRUE)  # Response vector (3 classes)
#' beta <- rnorm(15)  # Coefficient vector
#' alpha0 <- list(rep(0.5, 2), rep(0.3, 2))  # Initial alpha values for two profiles
#' pf.vec <- factor(sample(1:2, 10, replace = TRUE))  # Profile vector
#' omega <- 1  # Constraint for projection
#' L.step <- 0.01  # Learning step size
#' maxIter <- 100  # Maximum iterations
#' tol <- 1e-5  # Tolerance for convergence
#' loss.function <- "ce"  # Cross-entropy loss
#'
#' alpha <- alpha.compute(p, X, y, beta, alpha0, pf.vec, omega, L.step, maxIter, tol, loss.function)
#' print(alpha)
#'
#' @export
alpha.compute <- function(p, X, y, beta, alpha0, pf.vec, omega, L.step, maxIter, tol, loss.function) {
  tryCatch({
    # Number of sources (blocks)
    S <- length(p)

    alpha <- list()  # Initialize list to store alpha values for each profile

    # Iterate through profiles
    for (i in 1:length(levels(pf.vec))) {
      # Identify profile
      m <- as.integer(levels(pf.vec)[i])
      if (m == 0) {
        alpha[[i]] <- rep(0, S)  # Assign zero weights if profile is empty
        next
      }

      # Get samples associated with the current profile
      block.samples <- getBlockSamples(pf.vec, m, S)
      X.m <- X[block.samples$samples, ]

      if (loss.function == "ce") {
        K <- length(levels(as.factor(y)))  # Number of classes
        # Cross-entropy loss
        y.m <- y[block.samples$samples]  # Response for the current profile
        alpha[[i]] <- omega.norm.proj.method_ce(
          p, alpha0[[i]], X.m, y.m, beta, block.samples, K,
          omega, L.step, maxIter, tol
        )
      } else {
        # Other loss functions
        tilde.beta <- numeric()  # Prediction matrix
        col <- 1
        for (j in 1:S) {
          nextCol <- col + p[j] - 1
          if (j %in% block.samples$sources) {
            tilde.beta <- cbind(tilde.beta, X.m[, col:nextCol] %*% beta[col:nextCol])
          } else {
            tilde.beta <- cbind(tilde.beta, rep(0, nrow(X.m)))
          }
          col <- nextCol + 1
        }

        # Compute updated alpha values
        alpha[[i]] <- omega.norm.proj.method(
          y[block.samples$samples], alpha0[[i]], tilde.beta,
          omega, L.step, maxIter, tol, loss.function
        )
      }
    }

    return(alpha)
  }, error = function(e) {
    # In case of an error, return alpha0
    return(alpha0)
  })
}
