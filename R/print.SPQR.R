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
#' @return No return value, called for side effects.
#'
#' @examples
#' \donttest{
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' control <- list(iter = 200, warmup = 150, thin = 1)
#' fit <- SPQR(X = X, Y = Y, method = "MCMC", control = control,
#'             normalize = TRUE, verbose = FALSE)
#' print(fit, showModel = TRUE)
#' }
#' @export
print.SPQR <- function(x, ...) {
  s <- summary(x)
  print(s, ...)
}
