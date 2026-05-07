#' Alpha Initialization Function
#'
#' \code{alpha.initialization} is a function that initializes alpha values based on
#' the profile vector and user-defined parameters.
#'
#' @title Alpha Initialization Function
#'
#' @description
#' Initializes alpha values based on the profile vector and user-defined parameters.
#'
#' @param pf.vec Profile vector.
#' @param S Number of sources.
#' @param keep.alpha Logical indicating whether to keep alpha values the same for all profiles.
#' @param alpha0.comp Character specifying the method to compute alpha0 ("equiv" or "ones").
#'
#' @return
#' A list of initialized alpha values for each profile.
#'
#' @details
#' This function initializes alpha values based on the profile vector and user-defined
#' parameters. It can set alpha values uniformly across profiles or customize them
#' based on the profile's characteristics.
#'
#' @seealso
#' \code{\link{getBlockSamples}} for details on extracting block samples.
#'
#' @export
alpha.initialization = function(pf.vec, S, keep.alpha, alpha0.comp = "equiv"){
  # alpha0 weights
  alpha0 <- list()
  # Profiles
  profiles <- levels(pf.vec)

  # Initialize alpha
  if(alpha0.comp == "equiv"){
    if(keep.alpha){
      # All alpha's set to 1/n
      for(i in 1:length(profiles))
        alpha0[[i]] <- rep(1/length(pf.vec), S)
    } else {
      # All alpha's on profile set to 1/n_m (number of samples of each profile)
      for(i in 1:length(profiles)){
        # Profile m
        m <- as.integer(profiles[i])

        # Get block samples
        block.samples <- getBlockSamples(pf.vec, m, S)

        # Initialize alpha_m with 0's on the sources
        # that are not involved on the profile m
        alpha0.aux <- numeric(length = S)
        alpha0.aux[block.samples$sources] <- 1/length(block.samples$samples)
        alpha0[[i]] <- alpha0.aux
      }
    }
  } else {
    # All alpha's set to 1
    for(i in 1:length(profiles))
      alpha0[[i]] <- rep(1, S)
  }
  return(alpha0)
}
