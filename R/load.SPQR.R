#' @title load saved SPQR model
#' @description
#' Load saved SPQR model from a designated path. The function first loads
#' the \code{.SPQR} file that stores the \code{"SPQR"} object. It then checks
#' whether the SPQR model is fitted with \code{method = "MCMC"}. If not, it also
#' loads the \code{.pt} file storing the \code{torch} model with the same \code{name}
#' and attach it to the \code{"SPQR"} object.
#'
#' @name load.SPQR
#'
#' @param name The name of the saved object excluding extension.
#' @param path The path to look for the saved object. Default is the current working directory.
#'
#' @return An object of class \code{"SPQR"}.
#'
#' @examples
#' \dontrun{
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' fit <- SPQR(X = X, Y = Y, method = "MLE", normalize = TRUE)
#' save.SPQR(fit, name = "SPQR_MLE")
#' fit <- load.SPQR("SPQR_MLE")
#' }
#'
#' @export
load.SPQR <- function(name = stop("`name` must be specified"), path = NULL) {
  if (is.null(path)) path <- getwd()
  rds.name <- paste0(name, ".SPQR")
  object <- readRDS(file.path(path, rds.name))
  if (object$method != "MCMC") {
    pt.name <- paste0(name, ".pt")
    pt.model <- torch::torch_load(file.path(path, pt.name))
    object$model <- pt.model
  }
  return(object)
}
