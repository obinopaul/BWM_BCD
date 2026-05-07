#' Compute Lipschitz Constant for Alpha Optimization
#'
#' \code{compute_lipschitz_alpha.ce} calculates the Lipschitz constant for alpha optimization
#' based on the provided tilde beta values and feature matrix.
#'
#' @title Compute Lipschitz Constant for Alpha Optimization
#'
#' @description
#' This function computes the Lipschitz constant that is used in the alpha optimization process.
#' The Lipschitz constant is calculated based on the singular values of the feature matrix (X.m)
#' and is important for controlling the convergence behavior during optimization.
#'
#' @param X.m A numeric matrix containing the feature values for the profile.
#' @param n.m The number of samples corresponding to the current profile.
#'
#' @return
#' The Lipschitz constant for alpha optimization, which is the maximum of the calculated values
#' across all sources.
#'
#' @details
#' The function computes the Lipschitz constant by calculating the singular values of the input
#' feature matrix (\code{X.m}) using Singular Value Decomposition (SVD). The largest singular value
#' is squared and then divided by a factor of 4 times the number of samples (\code{n.m}). The function
#' returns the computed value.
#'
#' @examples
#' # Example usage:
#' X.m <- matrix(rnorm(50), nrow = 10, ncol = 5)  # Example feature matrix
#' n.m <- 10  # Number of samples
#' lipschitz_constant <- compute_lipschitz_alpha.ce(X.m, n.m)
#' print(lipschitz_constant)
#'
#' @export
compute_lipschitz_alpha.ce <- function(X.m, n.m) {
  # Identify columns with no missing values
  cols_to_keep <- apply(X.m, 2, function(col) !any(is.na(col)))

  # Subset the matrix X.m to keep only columns without missing values
  X.m_no_NAs <- X.m[, cols_to_keep]

  # Compute the largest singular value and square it
  max_singular_value <- max(svd(X.m_no_NAs)$d)^2

  # Calculate the Lipschitz constant
  lipschitz_constant <- max_singular_value / (4 * n.m)

  # Return the computed Lipschitz constant
  return(lipschitz_constant)
}
