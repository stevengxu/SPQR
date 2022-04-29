#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

#' @title autoplot method for class \code{SPQR}
#' @description
#' The function calls one of the following functions:
#' \code{\link[=plotEstimator]{plotEstimator()}}, \code{\link[=plotGOF]{plotGOF()}}, \code{\link[=plotMCMCtrace]{plotMCMCtrace()}},
#' \code{\link[=plotQALE]{plotQALE()}}, \code{\link[=plotQVI]{plotQVI()}}
#'
#' @method autoplot SPQR
#'
#' @param object An object of class \code{SPQR}.
#' @param output A character indicating the type of plot to be returned.
#'
#' \itemize{
#'   \item \code{"GOF"}: goodness of fit test by comparing the quantiles of probability integral transform (PIT) to that of uniform distribution.
#'   \item \code{"estimator"}: visualization of various estimates, including probability density function (PDF), cumulative density function (CDF) and quantile function (QF).
#'   \item \code{"trace"}: diagnostic trace plots for SPQR fitted with \code{method = "MCMC"}.
#'   \item \code{"QALE"}: quantile accumulative local effects (ALE) for visualizing covariate effects on predicted quantiles.
#'   \item \code{"QVI"}: quantile variable importance comparison.
#' }
#'
#' @param ... arguments passed into specific plot function, see \code{\link[=plotEstimator]{plotEstimator()}}, \code{\link[=plotGOF]{plotGOF()}}, \code{\link[=plotMCMCtrace]{plotMCMCtrace()}},
#' \code{\link[=plotQALE]{plotQALE()}} or \code{\link[=plotQVI]{plotQVI()}} for required arguments.
#'
#' @return a \code{ggplot} object
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
#'
#' ## Goodness-of-fit test
#' autoplot(fit, output = "GOF")
#' }
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
