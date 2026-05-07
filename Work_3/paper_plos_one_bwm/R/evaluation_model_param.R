#' Evaluate Model Performance Function
#'
#' \code{evaluation.model.param} evaluates the performance of predictive models for regression or classification tasks.
#'
#' @title Evaluate Model Performance Function
#'
#' @description
#' This function evaluates the performance of a predictive model using various evaluation metrics.
#' It supports both regression (\code{eval = "cont"}) and classification (\code{eval = "class"}) tasks.
#'
#' @param y.test Actual target values.
#' @param y.pred Predicted values from the model.
#' @param eval Type of evaluation: \code{"cont"} for continuous (regression) or \code{"class"} for classification. Default is \code{"cont"}.
#' @param n.vars Number of variables (optional, default is 0). Used only in regression to calculate Adjusted R-squared.
#'
#' @return
#' - For \code{eval = "cont"} (regression), a data frame with the following metrics:
#'   - Root Mean Square Error (RMSE)
#'   - Mean Absolute Error (MAE)
#'   - Mean Absolute Percentage Error (MAPE) or Symmetric Mean Absolute Percentage Error (SMAPE)
#'   - R-squared
#'   - Adjusted R-squared
#' - For \code{eval = "class"} (classification), a list containing:
#'   - \code{Confusion_Matrix}: A contingency table of observed vs. predicted classes.
#'   - \code{Metrics}: A data frame with:
#'     - Accuracy
#'     - Balanced Accuracy
#'     - Matthews Correlation Coefficient (MCC)
#'   - \code{Precision_Recall}: A data frame with precision, recall, and F1-score for each class.
#'
#' @details
#' - **Regression Metrics**:
#'   - RMSE and MAE are error-based metrics to measure prediction accuracy.
#'   - (S)MAPE handles cases with zero target values.
#'   - R-squared and Adjusted R-squared measure goodness-of-fit.
#' - **Classification Metrics**:
#'   - Accuracy indicates the proportion of correct predictions.
#'   - Balanced Accuracy handles imbalanced datasets by averaging recall across classes.
#'   - Precision, Recall, and F1-score provide class-specific performance measures.
#'   - MCC is a robust metric for both binary and multi-class problems.
#'
#' @examples
#' # Regression Example
#' y.test <- c(3.0, 2.5, 4.0)
#' y.pred <- c(2.8, 2.7, 4.1)
#' evaluation.model.param(y.test, y.pred, eval = "cont")
#'
#' # Classification Example
#' y.test <- c("A", "B", "A", "B", "A")
#' y.pred <- c("A", "B", "B", "B", "A")
#' evaluation.model.param(y.test, y.pred, eval = "class")
#'
#' @export
evaluation.model.param = function(y.test, y.pred, eval = "cont", n.vars = 0){
  if(eval == "cont"){
    # Convert factor to numeric
    y.test <- as.numeric(y.test)
    y.pred <- as.numeric(y.pred)

    # Number of samples
    n <- length(y.test)

    # Error term (y - predictions)
    error <- y.test - y.pred

    # Compute mean square error
    mean.sq.error <- sum(error^2)/n

    # Compute root mean square error
    root.mean.sq.error <- sqrt(mean.sq.error)

    # Compute mean absolute error
    mean.abs.error <- sum(abs(error))/n

    # Compute absolute percentage error
    absolute_percentage_error <- abs(error / y.test) * 100

    # Compute mean absolute percentage error
    mean.abs.per.error <- mean(absolute_percentage_error)
    mape <- TRUE

    if(mean.abs.per.error == Inf){
      # Compute symmetric mean absolute percentage error
      sym.mean.abs.per.error <- 2 * mean(abs(error) / (abs(y.test) + abs(y.pred))) * 100
      mape <- FALSE
    }

    # Compute R squared
    SS.res <- sum(error^2)
    mean.y <- mean(y.test)
    SS.tot <- sum((y.test - mean.y)^2)
    R.squared <- 1 - SS.res/SS.tot

    # Compute adjusted R squared
    adj.R.squared <- 1 - (SS.res*(n - 1))/(SS.tot*(n - n.vars - 1))

    # Evaluation parameters
    if(mape){
      evaluation_param <- data.frame(root.mean.sq.error, mean.abs.error, mean.abs.per.error, R.squared, adj.R.squared)
      colnames(evaluation_param) <- c("RMSE", "MAE", "MAPE", "R squared", "Adjusted R squared")
    } else {
      evaluation_param <- data.frame(root.mean.sq.error, mean.abs.error, sym.mean.abs.per.error, R.squared, adj.R.squared)
      colnames(evaluation_param) <- c("RMSE", "MAE", "SMAPE", "R squared", "Adjusted R squared")
    }

    return(evaluation_param)
  } else if(eval == "class") {
    # Ensure input is a factor
    y.test <- as.factor(y.test)
    y.pred <- as.factor(y.pred)

    # Compute confusion matrix
    conf.matrix <- table(y.test, y.pred)

    # Accuracy
    accuracy <- sum(diag(conf.matrix)) / sum(conf.matrix)

    # Precision, Recall, and F1-Score
    precision <- diag(conf.matrix) / rowSums(conf.matrix)
    recall <- diag(conf.matrix) / colSums(conf.matrix)
    f1_score <- 2 * (precision * recall) / (precision + recall)

    # Handle NaN in F1 scores (when precision and recall are 0)
    f1_score[is.nan(f1_score)] <- 0

    # Compute balanced accuracy (average recall)
    balanced_accuracy <- mean(recall, na.rm = TRUE)

    # Matthews Correlation Coefficient (MCC)
    mcc <- ifelse(length(conf.matrix) > 2,
                  cor(as.numeric(y.test), as.numeric(y.pred), method = "pearson"),
                  (conf.matrix[1, 1] * conf.matrix[2, 2] - conf.matrix[1, 2] * conf.matrix[2, 1]) /
                    sqrt((sum(conf.matrix[1, ]) * sum(conf.matrix[2, ])) *
                           (sum(conf.matrix[, 1]) * sum(conf.matrix[, 2]))))

    # Convert results to data frame
    classification_metrics <- data.frame(accuracy = accuracy,
                                         balanced_accuracy = balanced_accuracy,
                                         MCC = mcc)
    precision_recall <- data.frame(Class = levels(y.test),
                                   Precision = precision,
                                   Recall = recall,
                                   F1_Score = f1_score)

    return(list(Confusion_Matrix = conf.matrix,
                Metrics = classification_metrics,
                Precision_Recall = precision_recall))
  }
}
