#' @title print method for SPQR
#' @description Summarizes and print the output produced by SPQR() in an organized way.
#' @details This is equivalent to the function call \code{print.summary.SPQR(summary.SPQR(object), ...)}.
#'
#' @method print SPQR
#'
#' @param x An object of class \code{SPQR}
#' @param showModel If \code{TRUE}, prints the detailed NN architecture by layer.
#' @param ... Other arguments.
#'
#' @export
print.SPQR <- function(x, showModel = FALSE, ...) {
  s <- summary(x)
  print(s, showModel = showModel)
}
