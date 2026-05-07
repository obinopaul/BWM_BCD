#' Cross-Entropy Loss Function Evaluation
#'
#' \code{f.ce} evaluates the cross-entropy loss for multi-class classification problems
#' based on the given parameters and observed data.
#'
#' @title Cross-Entropy Loss Function Evaluation
#'
#' @description
#' This function computes the cross-entropy loss for multi-class classification by
#' calculating softmax probabilities for each sample in the dataset and summing up
#' the negative log probabilities of the true class labels.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param alpha.m A numeric vector of alpha weights corresponding to the current profile.
#' @param X.m A numeric matrix of predictors for the current profile.
#' @param y.m A numeric or factor vector of true class labels for the current profile.
#' @param beta A numeric vector containing interleaved block-class coefficients for all sources and classes.
#' @param block.samples A list indicating the sample indices for the current profile.
#' @param K An integer specifying the total number of classes in the classification problem.
#'
#' @return
#' A numeric scalar representing the cross-entropy loss for the given data and parameters.
#'
#' @details
#' The cross-entropy loss is computed by:
#' 1. Calculating the softmax probabilities for each class using the linear combinations
#'    of features and coefficients, weighted by the \code{alpha.m} parameters.
#' 2. Summing the negative log probabilities of the true class labels for each sample.
#'
#' The function depends on \code{\link{sigma_function}}, which computes the weighted
#' sum of features and coefficients for a given class.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)  # Two sources with 2 and 3 variables each
#' alpha.m <- c(0.5, 0.5)  # Equal weights for two sources in the profile
#' X.m <- matrix(rnorm(20), nrow = 5, ncol = 5)  # Feature matrix (5 samples, 5 features)
#' y.m <- c(1, 2, 3, 1, 2)  # True class labels
#' beta <- rnorm(15)  # Randomly initialized coefficients for 3 classes and 5 features
#' block.samples <- list(samples = 1:5)  # All samples in the current block
#' K <- 3  # Three classes
#'
#' loss <- f.ce(p, alpha.m, X.m, y.m, beta, block.samples, K)
#' print(loss)
#'
#' @seealso
#' \code{\link{sigma_function}} for computing the weighted linear combination of features.
#'
#' @export
f.ce <- function(p, alpha.m, X.m, y.m, beta, block.samples, K) {
  # Initialize softmax result
  softmax_res <- 0

  # Iterate over each sample
  for (l in 1:length(y.m)) {
    # Compute numerator and denominator for softmax probabilities
    numerator <- exp(sigma_function(p, alpha.m, X.m[l, ], y.m[l], beta, K, block.samples))
    denominator <- 0
    for (k in 1:K)
      denominator <- denominator + exp(sigma_function(p, alpha.m, X.m[l, ], k, beta, K, block.samples))

    # Accumulate the negative log softmax probability of the true class
    softmax_res <- softmax_res + log(denominator / numerator)
  }

  # Return the total cross-entropy loss
  return(as.numeric(softmax_res))
}
