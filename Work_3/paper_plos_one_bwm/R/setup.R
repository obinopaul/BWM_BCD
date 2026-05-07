#' Load Required R Packages
#'
#' \code{setup} is a function that loads the necessary R packages for the execution of the main algorithm.
#'
#' @title Load Required R Packages
#'
#' @description
#' This function loads specific R packages that are required for the main algorithm to run successfully.
#'
#' @details
#' The function loads the following R packages: caret, rlang, MASS, corrplot, glmnet, binaryLogic, ade4, naniar, mvtnorm, devtools, factoextra, and nnet.
#'
#' @examples
#' # Example usage:
#' setup()
#'
#' @export
setup = function(verbose = TRUE){
  load_library <- function(library_name){
    if(verbose)
      print(paste("Loading library: ", library_name))
    # Check if the library is installed
    if (!(library_name %in% installed.packages()))
      # If not installed, install it
      suppressWarnings(install.packages(library_name))

    # Load the library
    suppressWarnings(library(library_name, character.only = TRUE))
  }

  suppressMessages(load_library("caret"))
  suppressMessages(load_library("rlang"))
  suppressMessages(load_library("MASS"))
  suppressMessages(load_library("corrplot"))
  suppressMessages(load_library("glmnet"))
  suppressMessages(load_library("binaryLogic"))
  suppressMessages(load_library("ade4"))
  suppressMessages(load_library("naniar"))
  suppressMessages(load_library("mvtnorm"))
  suppressMessages(load_library("devtools"))
  suppressMessages(load_library("factoextra"))
  suppressMessages(load_library("nnet"))
}
