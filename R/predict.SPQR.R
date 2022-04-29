#' @title predict method for class \code{SPQR}
#' @description
#' Computes the predicted values for different functions based on the fitted \code{"SPQR"} object.
#'
#' @method predict SPQR
#' @name predict.SPQR
#'
#' @param object An object of class \code{SPQR}.
#' @param X The covariate vector/matrix for which the predictions are computed.
#' @param Y The response vector for which the predictions are computed. Default is \code{NULL}
#'   indicating that a equi-distant grid vector on [0,1] of length \code{nY} is used.
#' @param nY An integer number indicating length of grid when \code{Y} is not specified. Default: 101.
#' @param type The function to be predicted; \code{"PDF"}: probability density function,
#'  \code{"CDF"}: cumulative distribution function, and \code{"QF"}: the quantile function (default).
#' @param tau The grid of quantiles for which the quantile function is computed. Default: \code{seq(0.1,0.9,0.1)}.
#' @param ci.level The credible level for computing the pointwise credible intervals. The
#'  default is 0 indicating no credible intervals should be computed.
#' @param getAll If \code{TRUE}, extracts all posterior samples of the prediction. Default: \code{FALSE}.
#' @param ... Other arguments.
#'
#' @return A named array containing all predicted values.
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
#' ## compute the estimated PDF of Y conditioned on X = 0
#' pdf <- predict(fit, type = "PDF", X = 0, Y = seq(0, 1, 0.01))
#' plot(seq(0, 1, 0.01), pdf, xlab = "Y", ylab = "Density")
#' }
#' @export

predict.SPQR <- function(object, X, Y = NULL, nY = 101, type = c("QF","PDF","CDF"),
                         tau = seq(0.1,0.9,0.1), ci.level = 0, getAll = FALSE, ...) {
  type <- match.arg(type)
  if (object$method != "MCMC") {
    ci.level <- 0
    getAll <- FALSE
  }
  stopifnot(is.numeric(ci.level))
  if (ci.level < 0 || ci.level >=1) stop("`ci.level` should be between 0 and 1")

  Y.normalize <- X.normalize <- !is.null(object$normalize)
  p <- ncol(object$X)
  if (NCOL(X) != p) {
    if (NROW(X) != p) stop("incompatible dimensions")
    else dim(X) <- c(1,length(X)) # treat vector as single observation
  } else if (p == 1){
    dim(X) <- c(length(X),1)
  }
  if (X.normalize) {
    X.range <- object$normalize$X
    for (j in 1:p) {
      X[,j] <- (X[,j] - X.range[1,j])/(diff(X.range[,j]))
    }
  }
  if (is.null(Y) || type == "QF") {
    Y <- seq(0,1,length.out=nY)
  } else {
    if(is.matrix(Y) && NCOL(Y) == 1) Y <- drop(Y)
    if (Y.normalize) {
      Y.range <- object$normalize$Y
      Y <- (Y - Y.range[1])/(diff(Y.range))
    }
    if (min(Y)<0 || max(Y)>1) stop("values of `Y` should be between 0 and 1")
  }

  B <- .basis(Y, K = object$config$n.knots, integral = (type != "PDF"))
  if (object$method == "MCMC") {
    X <- t(X)
    n <- ncol(X)
    nest <- if (type == "QF") length(tau) else length(Y)
    nnn <- length(object$model)
    if (getAll) {
      out <- array(dim=c(n,nest,nnn))
      for (i in 1:nnn) {
        coefs <- .coefs(object$model[[i]], X, object$config$activation)
        out[,,i] <- .predict.SPQR(Y,coefs,B,type,tau)
      }
      dimnames(out)[[3]] <- 1:nnn
      names(dimnames(out))[3] <- "Iteration"
    } else if (ci.level > 0) {
      .out <- array(dim=c(n,nest,nnn))
      for (i in 1:nnn) {
        coefs <- .coefs(object$model[[i]], X, object$config$activation)
        .out[,,i] <- .predict.SPQR(Y,coefs,B,type,tau)
      }
      out <- array(dim=c(n,nest,3))
      out[,,1] <- apply(.out,1:2,quantile,probs=(1-ci.level)/2)
      out[,,2] <- apply(.out,1:2,mean)
      out[,,3] <- apply(.out,1:2,quantile,probs=(1+ci.level)/2)
      dimnames(out)[[3]] <- c("lower.bound","mean","upper.bound")
      names(dimnames(out))[3] <- "CI"
    } else {
      coefs <- matrix(0,nrow=n,ncol=ncol(B))
      for (i in 1:nnn) {
        coefs <- coefs + .coefs(object$model[[i]],X,object$config$activation)/nnn
      }
      out <- .predict.SPQR(Y, coefs, B, type, tau)
    }
  } else {
    model <- object$model
    model$eval()
    if (object$method == "MAP" && object$control$use.GPU) {
      model$to(device="cuda")
      X <- torch::torch_tensor(X, device="cuda")
      coefs <- as.matrix(model(X)$output$to(device="cpu"))
    } else {
      X <- torch::torch_tensor(X)
      coefs <- as.matrix(model(X)$output)
    }
    out <- .predict.SPQR(Y, coefs, B, type, tau)
  }
  if (type == "QF") {
    colnames(out) <- paste0(tau*100, "%")
    if (anyNA(out)) {
      warning('Some extreme quantiles could not be calculated. ',
              'Please increase the range of `Y`.')
    }
  } else {
    dimnames(out)[1:2] <- list(NULL,NULL)
  }
  if (Y.normalize) {
    Y.range <- object$normalize$Y
    if (type == "PDF") out <- out / diff(Y.range)
    else if (type == "QF") out <- out * diff(Y.range) + Y.range[1]
  }
  if (length(dim(out)) == 2) out <- drop(out)
  if (length(dim(out)) >= 2) {
    if (type == "QF") names(dimnames(out))[1:2] <- c("X","tau")
    else names(dimnames(out))[1:2] <- c("X","Y")
  }
  return(out)
}

.predict.SPQR <- function(Y, coefs, basis, type, tau) {
  df <- tcrossprod(coefs,basis)
  if (type != "QF") return(df)
  qf <- matrix(nrow=nrow(df),ncol=length(tau))
  for (ii in 1:nrow(qf)) qf[ii,] <- stats::approx(df[ii,], Y, xout=tau, ties = list("ordered", min))$y
  return(qf)
}
