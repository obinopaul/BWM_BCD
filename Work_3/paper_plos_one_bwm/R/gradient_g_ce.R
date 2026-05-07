#' Gradient of Cross-Entropy Loss with Respect to Coefficients (Beta)
#'
#' \code{gradient.g.ce} calculates the gradient of the cross-entropy loss function
#' with respect to the coefficients (\eqn{\\beta}) in a multi-class classification setting.
#'
#' @title Gradient of Cross-Entropy Loss with Respect to Beta
#'
#' @description
#' This function computes the gradient of the cross-entropy loss for multi-class classification problems,
#' taking into account multiple profiles, sources, and alpha weights. The function iterates through
#' sources and profiles, computing the gradient for each coefficient in the \eqn{\\beta} vector using
#' the \code{sigma_function} and a softmax formulation.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param X A feature matrix where rows correspond to samples and columns to features.
#' @param y A vector of observed class labels for the dataset.
#' @param alpha A list of numeric vectors, where each vector contains the alpha weights
#'   for the sources associated with a specific profile.
#' @param beta A numeric vector of coefficients, structured by interleaving blocks for sources and classes.
#' @param pf.vec A factor vector specifying the profile assignment for each sample in the dataset.
#'
#' @return
#' A numeric vector containing the gradient of the cross-entropy loss with respect to
#' each coefficient in the \eqn{\\beta} vector.
#'
#' @details
#' The gradient computation involves iterating over:
#' - **Sources**: Each source corresponds to a block of variables defined in \code{p}.
#' - **Profiles**: Observations are grouped by their profiles (\code{pf.vec}), and the alpha weights
#'   are applied accordingly.
#' - **Classes**: For each class, the gradient of the softmax function is computed using the difference
#'   between predicted probabilities and observed outcomes.
#'
#' For each sample, the contribution to the gradient is scaled by the alpha weight of the corresponding
#' source and normalized by the number of samples in its profile.
#'
#' @seealso
#' \code{\link{sigma_function}} for calculating the weighted linear combination of features and coefficients.
#' \code{\link{objective.fun.ce}} for the corresponding cross-entropy loss computation.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)
#' X <- matrix(rnorm(20), nrow = 5, ncol = 5)  # Feature matrix (5 samples, 5 features)
#' y <- factor(c(1, 2, 3, 1, 2))  # Observed classes
#' alpha <- list(c(0.5, 0.5), c(0.3, 0.7))  # Alpha weights for profiles
#' beta <- rnorm(15)  # Coefficient vector (5 variables * 3 classes)
#' pf.vec <- factor(c(1, 1, 2, 2, 1))  # Profile assignments
#'
#' grad <- gradient.g.ce(p, X, y, alpha, beta, pf.vec)
#' print(grad)
#'
#' @export
gradient.g.ce = function(p, X, y, alpha, beta, pf.vec){
  # Number of sources
  S <- length(p)

  # Profiles
  profiles <- levels(pf.vec)

  # Number of classes
  K <- length(levels(as.factor(y)))

  # Gradient vector
  grad.vec <- numeric(length = length(beta))
  col.source <- 1
  col.beta <- 1
  for(j.source in 1:S){
    next.col.source <- col.source + p[j.source] - 1
    next.col.beta <- col.beta + p[j.source]*K - 1
    for(r in 1:K){
      n_range <- (col.beta + p[j.source] * (r - 1)):(col.beta + p[j.source] * r - 1)
      # Initialize gradient value
      gradient <- numeric(length = p[j.source])
      for(i in 1:length(profiles)){
        # Profile m
        m <- as.integer(profiles[i])
        # Check if the source is on this profile
        if(!as.binary(m, n = S)[j.source])
          next;

        # Profile m alpha weights
        alpha.m <- alpha[[i]]

        # Samples with current profile
        block.samples <- getBlockSamples(pf.vec, m, S)
        X.m <- X[block.samples$samples,]

        # Outcome with current profile
        y.m <- y[block.samples$samples]

        for(l in 1:length(y.m)){
          numerator <- exp(alpha.m[j.source]*X.m[l,col.source:next.col.source]%*%beta[n_range])
          denominator <- 0
          for(k in 1:K)
            denominator <- denominator + exp(sigma_function(p, alpha.m, X.m[l,], k, beta, K, block.samples))

          # Vector from X.m is properly extracted
          x_vector <- as.vector(X.m[l, col.source:next.col.source])
          # Compute scalar operations explicitly
          substract <- if(y[l] == r) 1 else 0
          scalar_part <- as.numeric((numerator/denominator - substract) * alpha.m[j.source])
          # Ensure all dimensions match
          gradient <- gradient + scalar_part * x_vector
        }

        gradient <- gradient/length(y.m)
      }
      grad.vec[n_range] <- gradient
    }
    col.source <- next.col.source + 1
    col.beta <- next.col.beta + 1
  }

  return(grad.vec/length(profiles))
}
