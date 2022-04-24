#' @title coef method for class `SPQR`
#' @description
#' Computes the estimated spline coefficients of a `SPQR` class object
#' @name coef.SPQR
#'
#' @method coef SPQR
#'
#' @param object An object of class \code{SPQR}.
#' @param X The covariate vector/matrix for which the coefficient is calculated.
#' @param ... Other arguments.
#'
#' @return A \code{NROW(X)} by K matrix containing values of the estimated coefficient, where K is the number of basis functions.
#'
#' @examples
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' control <- list(iter = 300, warmup = 200, thin = 1)
#' fit <- SPQR(X = X, Y = Y, method = "MCMC", control = control, normalize = TRUE)
#' coef(fit, X = 0)
#'
#' @export
coef.SPQR <- function(object, X, ...) {

  p <- ncol(object$X)
  if (NCOL(X) != p) {
    if (NROW(X) != p) stop("incompatible dimensions")
    else dim(X) <- c(1,length(X)) # treat vector as single observation
  }
  if (!is.null(object$normalize)) {
    X.range <- object$normalize$X
    for (j in 1:p) {
      X[,p] <- (X[,p] - X.range[1,p])/(diff(X.range[,p]))
    }
  }
  if (object$method == "MCMC") {
    X <- t(X)
    n <- ncol(X)
    nnn <- length(object$model)
    out <- rowMeans(sapply(object$model, function(W){
      .coefs(W,X,object$config$activation)
    }))
    dim(out) <- c(n, object$config$n.knots)
  } else {
    model <- object$model
    model$eval()
    if (object$method == "MAP" && object$control$use.GPU) {
      model$to(device="cuda")
      X <- torch_tensor(X, device="cuda")
      out <- as.matrix(model(X)$output$to(device="cpu"))
    } else {
      X <- torch_tensor(X)
      out <- as.matrix(model(X)$output)
    }
  }
  colnames(out) <- paste0("theta[",1:object$config$n.knots,"]")
  names(dimnames(out)) <- c("X","Coefs")
  return(out)
}
