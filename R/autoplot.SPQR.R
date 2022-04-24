#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

#' @title autoplot method for class \code{"SPQR"}
#' @description
#' The function calls one of the following functions:
#' \code{plotEstimator()}, \code{plotGOF()}, \code{plotMCMCtrace()},
#' \code{plotQALE()}, \code{plotQVI()}
#'
#' @method autoplot SPQR
#'
#' @param object An object of class \code{"SPQR"}.
#' @param output A character indicating the type of plot to be returned.
#'
#' \itemize{
#'   \item "GOF": goodness of fit test by comparing the quantiles of probability integral transform (PIT) to that of uniform distribution.
#'   \item "estimator": visualization of various estimates, including probability density function (PDF), cumulative density function (CDF) and quantile function (QF).
#'   \item "trace": diagnostic trace plots for SPQR fitted with \code{method = "MCMC"}.
#'   \item "QALE": quantile accumulative local effects (ALE) for visualizing covariate effects on predicted quantiles.
#'   \item "QVI": quantile variable importance comparison.
#' }
#'
#' @param ... other arguments, see functions \code{plotEstimator()}, \code{plotGOF()}, \code{plotMCMCtrace()},
#' \code{plotQALE()} or \code{plotQVI()} for available arguments.
#'
#' @return a \code{"ggplot"} object
#' @export
autoplot.SPQR <- function(object, output=c("GOF","estimator","trace","QALE","QVI"), ...) {
  output <- match.arg(output)
  if (output == "estimator") {
    plotEstimator(object, ...)
  } else if (output == "GOF") {
    plotGOF(object, ...)
  } else if (output == "trace") {
    plotMCMCtrace(object, ...)
  } else if (output == "QALE") {
    plotQALE(object, ...)
  } else if (output == "QVI") {
    plotQVI(object, ...)
  }
}
