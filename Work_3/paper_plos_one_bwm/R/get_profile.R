#' Generate Profile Vector from Data Matrix
#'
#' \code{get_profile} is a function that generates a profile vector from a data matrix based on the provided
#' source vector.
#'
#' @title Generate Profile Vector from Data Matrix
#'
#' @description
#' Generates a profile vector from a data matrix based on the provided source vector. Each profile is
#' represented as a binary number in the profile vector.
#'
#' @param p Source vector representing the number of variables in each source.
#' @param X Data matrix.
#' @param showTable Logical, whether to print the profile matrix (default is FALSE).
#' @param saveTable Logical, whether to save the profile matrix as a CSV file (default is FALSE).
#'
#' @return
#' A profile vector representing profiles for each sample in the data matrix.
#'
#' @details
#' This function generates a profile vector based on the provided source vector and data matrix. The profile
#' vector represents profiles for each sample in the data matrix using binary numbers. Each binary digit
#' corresponds to whether a variable is present in a profile.
#'
#' @seealso
#' \code{\link{get_profile}} for details on how profiles are generated.
#'
#' @export
get_profile = function(p, X, showTable = F, saveTable = F){
  # Check dimensionality of source vector and matrix X
  if(sum(p) != dim(X)[2]){
    print("The number of variables of the data X do not concide with the provided source vector p.")
    return(NULL)
  }

  # Samples and Sources
  n <- dim(X)[1]
  S <- length(p)

  # Profile matrix
  pf_mat <- matrix(nrow = n, ncol = S + 1)

  # Profile vector
  pf.vec <- numeric(length = n)
  for(i in 1:n){
    # Profile of i-th sample
    pf <- 0
    col <- 1
    for(j in 1:S){
      nextCol <- col + p[j]
      if(!any(is.na(X[i, col:(nextCol - 1)]))){
        pf <- pf + 2^(S - j)
        pf_mat[i, j] = 1
      } else {
        pf_mat[i, j] = 0
      }

      col <- nextCol
    }

    # Add the i-th profile to the profile vector
    pf.vec[i] <- pf
    pf_mat[i, S + 1] <- pf
  }

  pf_df <- as.data.frame(pf_mat)
  colnames(pf_df) <- c(paste0("Source ", 1:S), "Profile")
  if(showTable)
    print(pf_df)

  if(saveTable)
    write.csv(pf_df, file = "bwm_pf_tab.csv", row.names = FALSE)

  return(as.factor(pf.vec))
}
