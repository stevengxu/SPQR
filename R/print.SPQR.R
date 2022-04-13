#' @title print method for SPQR
#' @method print SPQR
#'
#' @param x an object of class \code{SPQR}
#' @param showModel whether to print the NN structure
#' @param ... other arguments
#'
#' @export
print.SPQR <- function(x, showModel = FALSE, ...) {
  s <- summary(x)
  print(s, showModel = showModel)
}
