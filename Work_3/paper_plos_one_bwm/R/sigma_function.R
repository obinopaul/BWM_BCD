#' Sigma Function for Weighted Linear Combination
#'
#' \code{sigma_function} computes the weighted linear combination of features and coefficients
#' for a given class in a multi-class classification problem.
#'
#' @title Sigma Function for Weighted Linear Combination
#'
#' @description
#' This function calculates the weighted linear combination of features and coefficients
#' for a specific class in a multi-class classification model. It is used as a helper function
#' in functions like \code{f.ce} to evaluate probabilities or losses.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param alpha Alpha weights for the sources.
#' @param X Feature vector for a specific observation.
#' @param beta Coefficient vector containing interleaved block-class coefficients.
#' @param y Class index for which the linear combination is being computed.
#' @param K Number of classes for modeling.
#' @param block.samples A list containing the sources that are active, with an element named \code{sources}.
#'   The \code{sources} should be a vector of integers indicating the active blocks.
#'
#' @return
#' The weighted linear combination value for the specified class and observation.
#'
#' @details
#' The function iterates through the feature blocks and their corresponding alpha weights
#' to compute the contribution of each block. Coefficients for the specific class are extracted
#' from the interleaved \code{beta} vector based on the class index \code{y}.
#'
#' Only blocks that are active (as specified in \code{block.samples}) contribute to the
#' weighted combination.
#'
#' @seealso
#' \code{\link{f.ce}} for an example of how this function is used in cross-entropy loss evaluation.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)
#' alpha <- c(0.5, 0.5)
#' X <- c(1.2, -0.8, 0.5, 1.1, -0.3)
#' beta <- c(0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5)
#' y <- 2
#' K <- 2
#' block.samples <- list(sources = c(1, 2))
#' sigma_function(p, alpha, X, y, beta, K, block.samples)
#'
#' @export
sigma_function <- function(p, alpha, X, y, beta, K, block.samples) {
  # Number of sources
  S <- length(p)
  y <- as.numeric(y)  # Ensure class index is numeric
  val <- 0  # Initialize the result

  col <- 1  # Start column index for X
  betaCol <- 1  # Start column index for beta

  for (j in 1:S) {
    nextCol <- col + p[j] - 1  # End column for current block in X
    nextBetaCol <- betaCol + p[j] * K - 1  # End column for current block in beta

    # Check if block is active
    if (j %in% block.samples$sources) {
      # Indices for coefficients of class y in block j
      n_range <- (betaCol + p[j] * (y - 1)):(betaCol + p[j] * y - 1)

      # Weighted linear contribution from block j
      val <- val + alpha[j] * sum(X[col:nextCol] * beta[n_range])
    }

    # Update indices for the next block
    col <- nextCol + 1
    betaCol <- nextBetaCol + 1
  }

  return(as.numeric(val))  # Return result as numeric scalar
}
