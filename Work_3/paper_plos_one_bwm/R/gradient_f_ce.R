#' Gradient of Cross-Entropy Loss with Respect to Alpha
#'
#' \code{gradient.f.ce} computes the gradient of the cross-entropy loss function
#' with respect to the alpha values \eqn{\\alpha_m} for a specific profile.
#'
#' @title Gradient of Cross-Entropy Loss with Respect to Alpha
#'
#' @description
#' This function calculates the gradient of the cross-entropy loss function with respect to
#' the alpha weights \eqn{\\alpha_m}, which represent the importance of each source (block) in
#' multi-class classification. The gradient is computed by summing the contributions of all
#' samples across all blocks, using the softmax probabilities.
#'
#' @param p A vector specifying the number of variables for each source (block).
#' @param alpha.m A numeric vector of alpha weights corresponding to the current profile.
#' @param X.m A numeric matrix of predictors for the current profile.
#' @param beta A numeric vector containing interleaved block-class coefficients for all sources and classes.
#' @param y.m A numeric or factor vector of true class labels for the current profile.
#' @param block.samples A list indicating the sample indices for the current profile.
#' @param K An integer specifying the total number of classes in the classification problem.
#'
#' @return
#' A numeric vector containing the gradient of the cross-entropy loss with respect to
#' each alpha value \eqn{\\alpha_m} for the given profile.
#'
#' @details
#' This function iterates through each source (block) and computes the gradient of the cross-entropy loss
#' with respect to the corresponding alpha weight. For each sample:
#' 1. The softmax numerator and denominator are calculated across all classes.
#' 2. The contribution of the true class and other classes is used to update the gradient for the block.
#'
#' The \code{sigma_function} helper function computes the weighted linear combination of features
#' and coefficients for a specific class.
#'
#' @seealso
#' \code{\link{f.ce}} for the cross-entropy loss evaluation function.
#' \code{\link{sigma_function}} for the helper function that computes weighted sums.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)  # Two sources with 2 and 3 variables each
#' alpha.m <- c(0.5, 0.5)  # Equal weights for two sources in the profile
#' X.m <- matrix(rnorm(20), nrow = 5, ncol = 5)  # Feature matrix (5 samples, 5 features)
#' beta <- rnorm(15)  # Randomly initialized coefficients for 3 classes and 5 features
#' y.m <- c(1, 2, 3, 1, 2)  # True class labels
#' block.samples <- list(samples = 1:5)  # All samples in the current block
#' K <- 3  # Three classes
#'
#' gradient <- gradient.f.ce(p, alpha.m, X.m, y.m, beta, block.samples, K)
#' print(gradient)
#'
#' @export
gradient.f.ce <- function(p, alpha.m, X.m, y.m, beta, block.samples, K) {
  S <- length(p)  # Number of sources (blocks)
  grad_alpha.m <- numeric(length = S)  # Initialize gradient vector
  col <- 1  # Starting column index for features
  betaCol <- 1  # Starting index for beta coefficients

  # Iterate through each source (block)
  for (j in 1:S) {
    nextCol <- col + p[j] - 1  # Ending column index for the current block
    nextBetaCol <- betaCol + p[j] * K - 1  # Ending index for the current block's coefficients

    res <- 0  # Initialize the gradient contribution for the current block
    if(j %in% block.samples$sources){
      for (l in 1:length(y.m)) {  # Iterate through each sample
        # Compute numerator and denominator for the softmax gradient
        numerator <- 0
        denominator <- 0
        for (k in 1:K) {
          sigma_val_aux <- exp(sigma_function(p, alpha.m, X.m[l, ], k, beta, K, block.samples))
          n_range <- (betaCol + p[j] * (k - 1)):(betaCol + p[j] * k - 1)  # Coefficient indices for class k
          numerator <- numerator + as.numeric(sigma_val_aux * X.m[l, col:nextCol] %*% beta[n_range])
          denominator <- denominator + sigma_val_aux
        }

        # True class coefficient range
        val.y <- as.numeric(y.m[l])  # True class label for the sample
        n_range <- (betaCol + p[j] * (val.y - 1)):(betaCol + p[j] * val.y - 1)

        # Update gradient contribution for the current sample
        res <- res + numerator / denominator - X.m[l, col:nextCol] %*% beta[n_range]
      }
    }

    # Store the gradient for the current block
    grad_alpha.m[j] <- res

    # Update column and beta indices for the next block
    col <- nextCol + 1
    betaCol <- nextBetaCol + 1
  }

  return(grad_alpha.m)
}
