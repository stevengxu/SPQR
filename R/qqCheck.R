#' @title Goodness-of-fit test for SPQR estimator
#' @description
#' Performs a goodness-of-fit test for the estimated conditional probability density function (PDF) using
#' inverse transformation method.
#'
#' @param object An object of class \code{"SPQR"}.
#' @param ci.level The credible level for plotting the credible bands for the Q-Q line when \code{object$method == "MCMC"}
#'   is fitted using \code{method="MCMC"}. The default is 0 indicating no credible bands should be plotted.
#' @param getAll If \code{TRUE} and \code{object$method == "MCMC"}, plots all posterior samples of Q-Q lines. Default: \code{FALSE}.
#'
#' @return A \code{ggplot} object.
#'
#' @import ggplot2
#'
#' @export

qqCheck <- function(object, ci.level = 0, getAll = FALSE) {

  stopifnot(is.numeric(ci.level))
  if (ci.level < 0 || ci.level >=1) stop("`ci.level` should be between 0 and 1")

  B <- .basis(object$Y, K = object$config$n.knots, integral = TRUE)
  qu <- stats::ppoints(max(1e3,length(object$Y)))
  if (object$method == "MCMC") {
    X <- t(object$X)
    n <- ncol(X)
    if (getAll) {
      cdf <- sapply(object$model,function(W){
        coefs <- .coefs(W, X, object$config$activation)
        rowSums(coefs*B)})
      .qqplot(qu,cdf,getAll=TRUE)
    } else if (ci.level > 0) {
      .cdf <- sapply(object$model,function(W){
        coefs <- .coefs(W, X, object$config$activation)
        rowSums(coefs*B)})
      cdf <- matrix(nrow=nrow(.cdf),ncol=3)
      cdf[,1] <- apply(.cdf,1,quantile,probs=(1-ci.level)/2)
      cdf[,2] <- apply(.cdf,1,mean)
      cdf[,3] <- apply(.cdf,1,quantile,probs=(1+ci.level)/2)
      .qqplot(qu,cdf,ci.level=ci.level)
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

.qqplot <- function(x, y, ci.level = 0, getAll = FALSE){

  lenx <- length(x)
  sx <- sort(x)
  leny <- NROW(y)
  if (leny < lenx) sx <- stats::approx(1L:lenx, sx, n = leny)$y
  if (getAll) {
    sy <- apply(y,2,sort)
    my <- sort(rowMeans(y))
    dat <- data.frame(sx=sx, my=my)
    datt <- data.frame(sx=sx, sy=c(sy), g=rep(1:ncol(sy),each=nrow(sy)))
    p <-
      ggplot() +
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_point(data=dat, aes(x=.data$sx, y=.data$my), alpha=0.3, shape=19, size=2) +
      geom_line(data=datt, aes(x=.data$sx, y=.data$sy, group=.data$g), alpha=0.2)
  } else if (ci.level > 0){
    oy <- order(y[,2])
    sy <- y[oy,]
    dat <- data.frame(sx = sx, sy = sy[,2], ymin = sy[,1], ymax = sy[,3])
    p <-
      ggplot(data=dat, aes(x=.data$sx, y=.data$sy)) +
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_ribbon(aes(x=.data$sx, ymin=.data$ymin, ymax=.data$ymax), alpha=0.3) +
      geom_point(alpha=0.3, shape=19, size=2)
  } else {
    sy <- sort(y)
    dat <- data.frame(sx = sx, sy = sy)
    p <-
      ggplot(dat, aes(x=.data$sx, y=.data$sy)) +
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_point(alpha=0.3, shape=19, size=2)
  }
  p + theme_bw() + labs(title="Q-Q Plot", x="Unif(0,1)", y="Fitted CDF") +
    theme(axis.text = element_text(colour="black", size = 12),
          axis.title=element_text(size=15),
          plot.title=element_text(hjust=0.5, size=18))
}
