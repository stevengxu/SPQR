#' @title print method for class \code{SPQR}
#' @description Summarizes and print the output produced by SPQR() in an organized way.
#' @details This is equivalent to the function call \code{print.summary.SPQR(summary.SPQR(object), ...)}.
#'
#' @method print SPQR
#' @name print.SPQR
#'
#' @param x An object of class \code{SPQR}
#' @inheritDotParams print.summary.SPQR -x
#'
#' @examples
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' control <- list(iter = 300, warmup = 200, thin = 1)
#' fit <- SPQR(X = X, Y = Y, method = "MCMC", control = control, normalize = TRUE)
#' print(fit, showModel = TRUE)
#'
#' @export
print.SPQR <- function(x, ...) {
  s <- summary(x)
  print(s, ...)
}
