#' @title save fitted SPQR model
#' @description
#' Save \code{"SPQR"} object in a designated directory. If SPQR is fitted with
#' \code{method = "MCMC"} then only a \code{.SPQR} file is saved; otherwise, a \code{.pt}
#' file storing the fitted \code{torch} model is also saved.
#'
#' @name save.SPQR
#'
#' @param object An object of class \code{"SPQR"}.
#' @param name The name of the saved object excluding extension.
#' @param path The path to save the object. Default is the current working directory.
#'
#'
#' @examples
#' \dontrun{
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' fit <- SPQR(X = X, Y = Y, method = "MLE", normalize = TRUE)
#' save.SPQR(fit, name = "SPQR_MLE")
#' }
#'
#' @export
save.SPQR <- function(object, name = stop("`name` must be specified"), path = NULL) {
  if (is.null(path)) path <- getwd()
  rds.name <- paste0(name, ".SPQR")
  if (object$method != "MCMC") {
    pt.model <- object$model
    object$model <- NULL
    pt.name <- paste0(name, ".pt")
    torch::torch_save(pt.model, file.path(path, pt.name))
  }
  saveRDS(object, file=file.path(path, rds.name))
}
