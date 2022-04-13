#' @title coef method for class `SPQR`
#' @description
#' Computes the estimated spline coefficients of a `SPQR` class object
#' @name coef.SPQR
#'
#' @param object an object of class \code{SPQR}
#' @param newx covariate values for which the coefficient is calculated
#'
#'
#' @method coef SPQR
#' @export
coef.SPQR <- function(object, newx, ...) {

  p <- ncol(object$X)
  if (NCOL(newx) != p) {
    if (NROW(newx) != p) stop("incompatible dimensions")
    else dim(newx) <- c(1,length(newx)) # treat vector as single observation
  }
  if (!is.null(object$normalize)) {
    X.range <- object$normalize$X
    for (j in 1:p) {
      newx[,p] <- (newx[,p] - X.range[1,p])/(diff(X.range[,p]))
    }
  }
  if (object$method == "MCMC") {
    newx <- t(newx)
    n <- ncol(newx)
    nnn <- length(object$model)
    out <- rowMeans(sapply(object$model, function(W){
      .coefs(W,newx,object$config$activation)
    }))
    dim(out) <- c(n, object$config$n.knots)
  } else {
    model <- object$model
    model$eval()
    if (object$method == "MAP" && object$control$use.GPU) {
      model$to(device="cuda")
      newx <- torch_tensor(newx, device="cuda")
      out <- as.matrix(model(newx)$output$to(device="cpu"))
    } else {
      newx <- torch_tensor(newx)
      out <- as.matrix(model(newx)$output)
    }
  }
  colnames(out) <- paste0("theta[",1:object$config$n.knots,"]")
  names(dimnames(out)) <- c("X","Coefs")
  return(out)
}
