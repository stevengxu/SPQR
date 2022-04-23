#' @title print method for SPQR
#' @description Summarizes and print the output produced by SPQR() in an organized way.
#' @details This is equivalent to the function call \code{print.summary.SPQR(summary.SPQR(object), ...)}.
#'
#' @method print SPQR
#'
#' @param x An object of class \code{SPQR}
#' @inheritDotParams print.summary.SPQR -x
#'
#' @export
print.SPQR <- function(x, ...) {
  s <- summary(x)
  print(s, ...)
}
