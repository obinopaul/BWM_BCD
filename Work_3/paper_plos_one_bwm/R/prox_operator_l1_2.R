#' Proximal Operator for L1/L2 Regularization
#'
#' \code{prox.operator.l1_l2} is a function that computes the proximal operator for L1/L2 regularization.
#'
#' @title Proximal Operator for L1/L2 Regularization
#'
#' @description
#' This function computes the proximal operator for L1/L2 regularization based on the given parameters.
#'
#' @param p A vector specifying the partition of the input vector u.
#' @param u The input vector.
#' @param mu The regularization parameter.
#'
#' @return
#' The computed proximal operator for L1/L2 regularization.
#'
#' @details
#' The proximal operator for L1/L2 regularization is used to perform regularization on groups of elements of the input vector u, where each group is defined by the partition vector p. It applies the element-wise proximal operator for L1 regularization within each group.
#'
#' @examples
#' # Example usage:
#' p <- c(2, 3, 2)
#' u <- c(0.5, -0.2, 0.7, 1.0, -1.5, 2.0, -0.3, 0.8)
#' mu <- 0.1
#' prox <- prox.operator.l1_l2(p, u, mu)
#'
#' @export
prox.operator.l1_l2 <- function(p, u, mu){
  if(length(u) != sum(p))
    return(u)

  # Optimal solution beta
  beta <- numeric(length = length(u))

  # Partition range
  group.init <- 1
  for(i in 1:length(p)){
    group.end <- group.init + p[i]
    group.range <- group.init:(group.end - 1)

    # Since the problem is separable, we compute the optimal
    # solution for each group
    l2.norm.u_group <- (sum(u[group.range]^2))^(1/2)
    beta[group.range] <- max((1 - sqrt(p[i])*mu/l2.norm.u_group), 0)*u[group.range]

    group.init <- group.end
  }

  return(beta)
}
