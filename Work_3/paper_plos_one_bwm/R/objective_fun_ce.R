#' Cross-Entropy Objective Function for Multi-Class Classification
#'
#' \code{objective.fun.ce} calculates the negative cross-entropy loss for a
#' multi-class classification model with block-specific weights and profiles.
#'
#' @title Cross-Entropy Objective Function
#'
#' @description
#' This function computes the average negative log-likelihood (cross-entropy loss)
#' for a multi-class classification problem. It accounts for different profiles,
#' sources, and their corresponding weights in the calculation.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param X A feature matrix where rows correspond to samples and columns to features.
#' @param y A factor vector containing the true class labels for each observation.
#' @param beta A numeric vector of coefficients for all features and classes, arranged
#'   in interleaved blocks.
#' @param alpha A list of numeric vectors, where each vector contains the alpha weights
#'   for the sources corresponding to a specific profile.
#' @param pf.vec A factor vector specifying the profile for each observation.
#'
#' @return
#' A numeric scalar representing the negative cross-entropy loss normalized by the
#' number of profiles.
#'
#' @details
#' The function iterates over the profiles defined in \code{pf.vec}. For each profile,
#' it selects the corresponding samples and computes the cross-entropy loss using a
#' softmax function. The softmax is computed based on the weighted linear combination
#' of features and coefficients for each class. This allows for profile- and source-specific
#' weighting of the features in the loss computation.
#'
#' The objective function is normalized by both the number of samples in each profile
#' and the total number of profiles.
#'
#' @examples
#' # Example data setup
#' p <- c(2, 3)
#' X <- matrix(runif(100), nrow = 10)  # Feature matrix (10 samples, 10 features)
#' y <- factor(sample(1:3, 10, replace = TRUE))  # Class labels (1 to 3)
#' beta <- runif(30)  # Coefficient vector (e.g., 10 features * 3 classes)
#' alpha <- list(c(0.5, 0.5), c(0.6, 0.4))  # Profile-specific alpha weights
#' pf.vec <- factor(sample(1:2, 10, replace = TRUE))  # Profile assignments
#'
#' # Compute the objective function
#' objective_value <- objective.fun.ce(p, X, y, beta, alpha, pf.vec)
#' print(objective_value)
#'
#' @seealso
#' \code{\link{sigma_function}} for the weighted linear combination function used in the softmax.
#'
#' @export
objective.fun.ce <- function(p, X, y, beta, alpha, pf.vec) {
  # Number of sources
  S <- length(p)
  # Profiles
  profiles <- levels(pf.vec)
  # Number of classes minus 1
  K <- length(levels(as.factor(y)))

  # Initialize the objective function value
  obj.func <- 0

  for (i in seq_along(profiles)) {
    # Current profile index
    m <- as.integer(profiles[i])

    # Alpha vector for the current profile
    alpha.m <- alpha[[i]]

    # Get block samples for the current profile
    block.samples <- getBlockSamples(pf.vec, m, S)
    X.m <- X[block.samples$samples, ]  # Keep as matrix if single row
    y.m <- y[block.samples$samples]

    # Compute the cross-entropy loss for the current profile
    softmax_res <- 0

    for (l in seq_along(y.m)) {
      # Compute the numerator and denominator of the softmax
      numerator <- exp(sigma_function(p, alpha.m, X.m[l, ], y.m[l], beta, K, block.samples)) # Softmax calculations
      denominator <- 0
      for(k in 1:K)
        denominator <- denominator + exp(sigma_function(p, alpha.m, X.m[l, ],
                                                        k, beta, K, block.samples
                                                        )) # Softmax calculations


      # Accumulate the log probability
      softmax_res <- softmax_res + log(numerator / denominator)
    }

    # Average over the current profile and update the total objective function
    obj.func <- obj.func + softmax_res / length(y.m)
  }

  # Normalize by the number of profiles and return the negative log likelihood
  return(as.numeric(-obj.func / length(profiles)))
}
