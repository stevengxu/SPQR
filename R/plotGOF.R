#' @title goodness-of-fit test for SPQR estimator
#' @description
#' Performs a goodness-of-fit test for the estimated conditional probability density function (PDF) using
#' probability inverse transformation method.
#' @name plotGOF
#'
#' @param object An object of class \code{SPQR}.
#' @param getAll If \code{TRUE} and SPQR is fitted with \code{method = "MCMC"}, plots all posterior samples of Q-Q lines. Default: \code{FALSE}.
#'
#' @return A \code{ggplot} object.
#'
#' @import ggplot2
#'
#' @examples
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' control <- list(iter = 200, warmup = 150, thin = 1)
#' fit <- SPQR(X = X, Y = Y, method = "MCMC", control = control,
#'             normalize = TRUE, verbose = FALSE)
#'
#' ## Goodness-of-fit test
#' plotGOF(fit)
#'
#' @export
plotGOF <- function(object, getAll = FALSE) {

  X <- object$X; Y <- object$Y
  p <- NCOL(X)
  normalize <- !is.null(object$normalize)
  if (normalize) {
    X.range <- object$normalize$X
    for (j in 1:p) {
      X[,j] <- (X[,j] - X.range[1,j])/(diff(X.range[,j]))
    }
    Y.range <- object$normalize$Y
    Y <- (Y - Y.range[1])/(diff(Y.range))
  }
  B <- .basis(Y, K = object$config$n.knots, integral = TRUE)
  qu <- stats::ppoints(max(1e3,length(Y)))
  if (object$method == "MCMC") {
    X <- t(X)
    n <- ncol(X)
    if (getAll) {
      cdf <- sapply(object$model,function(W){
        coefs <- .coefs(W, X, object$config$activation)
        rowSums(coefs*B)})
      .qqplot(qu,cdf,getAll=TRUE)
    } else {
      coefs <- rowMeans(sapply(object$model, function(W){
        .coefs(W,X,object$config$activation)
      }))
      dim(coefs) <- c(n, object$config$n.knots)
      cdf <- rowSums(coefs*B)
      .qqplot(qu,cdf)
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
    cdf <- rowSums(coefs*B)
    .qqplot(qu,cdf)
  }
}

.qqplot <- function(x, y, getAll = FALSE){

  lenx <- length(x)
  sx <- sort(x)
  leny <- NROW(y)
  if (leny < lenx) sx <- stats::approx(1L:lenx, sx, n = leny)$y
  if (getAll) {
    sy <- apply(y,2,sort)
    my <- rowMeans(sy)
    dat <- data.frame(sx=sx, my=my)
    datt <- data.frame(sx=sx, sy=c(sy), g=rep(1:ncol(sy),each=nrow(sy)))
    p <-
      ggplot() +
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_line(data=datt, aes(x=.data$sx, y=.data$sy, group=.data$g), alpha=0.1)
  } else {
    sy <- sort(y)
    dat <- data.frame(sx = sx, sy = sy)
    p <-
      ggplot(dat, aes(x=.data$sx, y=.data$sy)) +
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_point(shape=19, size=1.2)
  }
  p + theme_bw() + labs(title="Q-Q Plot", x="Theoretical Quantiles", y="Probability integral transform (PIT)") +
    theme(axis.text = element_text(colour="black", size = 12),
          axis.title=element_text(size=15),
          plot.title=element_text(hjust=0.5, size=18))
}
