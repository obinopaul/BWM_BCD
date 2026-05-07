#' Get Block Samples for a Given Profile
#'
#' \code{getBlockSamples} is a function that retrieves the samples and sources associated with a given profile.
#'
#' @title Get Block Samples for a Given Profile
#'
#' @description
#' Retrieves the samples and sources associated with a given profile from the profile vector.
#'
#' @param pf.vec Profile vector containing profiles for each sample.
#' @param m Profile of interest.
#' @param S Number of sources.
#'
#' @return
#' A list containing the samples and sources associated with the specified profile.
#'
#' @details
#' This function identifies and collects the samples and sources associated with a specific profile
#' from the profile vector. It returns the samples belonging to the profile and the sources that are
#' relevant to that profile.
#'
#' @export
getBlockSamples = function(pf.vec, m, S){
  # Get sources of the given profile
  sources.on.profile <- which(as.binary(m, n = S))

  # Set profiles
  profiles <- levels(pf.vec)

  # Add corresponding samples to the block
  samples.block <- numeric()
  for(i in 1:length(profiles)){
    profile <- as.integer(profiles[i])
    if(all(as.binary(profile, n = S)[sources.on.profile]))
      samples.block <-
        c(samples.block, which(pf.vec == profile))
  }

  # Return the block and the sources related to that block
  return(list(samples = samples.block,
              sources = sources.on.profile))
}
