#' Beta Initialization Function
#'
#' \code{beta.initialization} initializes the \eqn{\\beta} coefficients for a regression
#' model using various statistical methods based on the input data and user-defined parameters.
#'
#' @title Beta Initialization Function
#'
#' @description
#' Initializes the \eqn{\\beta} coefficients for a regression model based on the provided
#' data, profile structure, and initialization method. Supported methods include
#' robust linear regression, lasso regression, logistic regression, and multinomial logistic regression.
#'
#' @param p A vector specifying the number of variables for each source.
#' @param X A matrix of predictors, where rows correspond to samples and columns to features.
#' @param y A response vector (continuous or categorical) depending on the chosen initialization method.
#' @param beta0.comp A character string specifying the initialization method. Options include:
#'   - \code{"LMR"}: Linear Model Regression (Robust).
#'   - \code{"LR"}: Lasso Regression.
#'   - \code{"LogR"}: Logistic Regression.
#'   - \code{"MultinomR"}: Multinomial Logistic Regression.
#'
#' @return
#' A numeric vector of initialized \eqn{\\beta} coefficients, structured by sources and classes.
#'
#' @details
#' The function supports different initialization methods for \eqn{\\beta} coefficients:
#' - **Linear Model Regression (LMR):** Uses robust linear regression to mitigate the impact of outliers.
#' - **Lasso Regression (LR):** Uses L1-regularized regression for feature selection and coefficient shrinkage.
#' - **Logistic Regression (LogR):** Fits a logistic regression model for binary classification problems.
#' - **Multinomial Logistic Regression (MultinomR):** Fits a multinomial regression model for multi-class problems.
#'
#' The function iterates over sources (\code{p}) to compute coefficients independently for each block of variables.
#' Missing data in \code{X} is handled by excluding incomplete rows for each source.
#'
#' @seealso
#' \code{\link{rlm}} for robust linear regression,
#' \code{\link{cv.glmnet}} for Lasso regression,
#' \code{\link{glm}} for logistic regression,
#' \code{\link{multinom}} for multinomial logistic regression.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3)  # Two sources with 2 and 3 variables each
#' X <- matrix(rnorm(50), nrow = 10, ncol = 5)  # Feature matrix (10 samples, 5 features)
#' y <- factor(c(1, 2, 1, 3, 2, 1, 3, 3, 2, 1))  # Multi-class response
#' beta0.comp <- "MultinomR"  # Use multinomial regression for initialization
#'
#' beta.init <- beta.initialization(p, X, y, beta0.comp)
#' print(beta.init)
#'
#' @export
beta.initialization <- function(p, X, y, beta0.comp){
  # Number of sources
  S <- length(p)

  # beta0 initialization model
  beta0.compute <- switch (
    beta0.comp,

    # Linear Model Regression
    # We use a robust one for the presence of outliers
    "LMR" = function(X, y){
      return(as.vector(rlm(y ~ . + 0, data = data.frame(X))$coefficients))
    },

    # Lasso Regression
    "LR" = function(X, y){
      # Lasso (alpha = 1, lasso penalty)
      cv_lasso_model <- cv.glmnet(x = as.matrix(X), y = y, family = "gaussian",
                                  alpha = 1, intercept = F, nfolds = 5)

      # Best lambda value model
      lambda_lasso <- cv_lasso_model$lambda.min
      return(as.vector(glmnet(x = as.matrix(X), y = y, family = "gaussian", alpha = 1,
                              intercept = F, lambda = lambda_lasso)$beta[,1]))
    },

    # Logistic Regression
    "LogR" = function(X, y){
      regression_log_model <- suppressWarnings(glm(as.factor(y) ~ as.matrix(X),
                                                   family = 'binomial', TRACE = TRUE,
                                                   control = glm.control(epsilon = 1e-04,
                                                                         maxit = 1000)))
      coeff <- regression_log_model$coefficients[-1]
      if(any(abs(coeff) > 20))
        coeff <- coeff/max(abs(coeff[abs(coeff) > 20]), na.rm = TRUE)

      return(coeff)
    },

    # Multinomial Logistic Regression
    "MultinomR" = function(X, y){
      # Fit multinomial regression model
      X <- as.matrix(X)
      regression_multinom_model <- glmnet(X, y, family = "multinomial", alpha = 1)
      # Extract coefficients
      coefs <- lapply(coef(regression_multinom_model), as.matrix)
      coefs_total <- c()
      for(level in levels(y)){
        coefs_total <- c(coefs_total, rowMeans(coefs[[level]])[-1])
      }

      # Return the final vector of coefficients
      return(coefs_total)
    },

    return(NULL)
  )

  # Beta coefficients
  K <- 1
  if(beta0.comp != "LMR"){
    y <- as.factor(y)
    if(beta0.comp == "MultinomR")
      K <- length(levels(y))
  }
  beta.coeff <- numeric(length = dim(X)[2]*K)
  col <- 1
  colBeta <- 1
  for(i in 1:S){
    nextCol <- col + p[i] - 1
    nextColBeta <- colBeta + p[i]*K - 1

    # Samples in source i with complete data
    ind.samp <- rowSums(is.na(X[, col:nextCol])) == 0
    X.complete <- X[ind.samp, col:nextCol]
    y.complete <- y[ind.samp]
    if(is_empty(y.complete))
      next
    # Beta coefficient for source i
    beta0.coeff <- beta0.compute(X.complete, y.complete)
    beta.coeff[colBeta:nextColBeta] <- beta0.coeff
    beta.coeff[is.na(beta.coeff)] <- 0
    col <- nextCol + 1
    colBeta <- nextColBeta + 1
  }

  return(beta.coeff)
}
