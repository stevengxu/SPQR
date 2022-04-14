#' @title autoplot method for SPQR
#' @description
#' Computes and plots the estimated PDF/CDF/QF curves.
#'
#' @method autoplot SPQR
#'
#' @param object An object of class \code{SPQR}
#' @param X A row vector indicating covariate values for which the conditional PDF/CDF/QF is computed and plotted.
#' @param nY An integer number indicating length of grid for which the PDF/CDF/QF is computed. Default: 101.
#' @param type The function to be plotted; \code{"PDF"}: probability density function,
#'  \code{"CDF"}: cumulative distribution function, and \code{"QF"}: the quantile function (default).
#' @param tau The grid of quantiles for which the quantile function is computed.
#' @param ci.level The credible level for plotting the credible bands. The default is 0 indicating no credible bands should be plotted.
#' @param getAll If \code{TRUE}, all posterior samples of the curve should be plotted. Default: \code{FALSE}.
#' @param ... other arguments.
#'
#' @return A \code{ggplot} object.
#'
#' @import ggplot2
#'
#' @export
autoplot.SPQR <- function(object, X, nY=501, type = c("QF","PDF","CDF"),
tau = seq(0.05,0.95,0.05), ci.level = 0, getAll = FALSE, ...) {
  type <- match.arg(type)
  if (NROW(X) > 1 && NCOL(X) > 1)
    stop("`X` should contain exactly 1 observation")
  if (object$method != "MCMC") {
    ci.level <- 0
    getAll <- FALSE
  }
  yy <- predict.SPQR(object, X=X, nY=nY, type=type, tau=tau, ci.level=ci.level, getAll=getAll)
  if (type != "QF") {
    # need Y on original scale for plotting
    Y <- seq(0,1,length.out=nY)
    Y.normalize <- !is.null(object$normalize)
    if (Y.normalize) {
      Y.range <- object$normalize$Y
      Y <- Y*diff(Y.range) + Y.range[1]
    }
  }
  xx <- if (type=="QF") tau else Y
  if (getAll) {
    nnn <- length(object$model)
    ym <- as.vector(apply(yy,1:2,mean))
    dfm <- data.frame(x=xx, y=ym)
    dfa <- data.frame(x=rep(xx,nnn),y=as.vector(yy),g=rep(seq_len(nnn),each=length(xx)))
    p <-
      ggplot() +
      geom_line(data=dfa, aes(x=.data$x,y=.data$y,group=.data$g), alpha=0.2) +
      geom_line(data=dfm, aes(x=.data$x,y=.data$y), size=1, color="red")
  } else if (ci.level > 0) {
    df <- data.frame(x=xx, y=yy[,,2], ymin=yy[,,1], ymax=yy[,,3])
    p <-
      ggplot(data=df, aes(x=.data$x, y=.data$y)) +
      geom_line(size=1) +
      geom_ribbon(aes(x=.data$x, ymin=.data$ymin, ymax=.data$ymax), alpha=0.2)
  } else {
    df <- data.frame(x=xx, y=yy)
    p <-
      ggplot(data=df, aes(x=.data$x,y=.data$y)) +
      geom_line(size=1)
  }
  if (type == "QF") {
    p <- p + labs(x=parse(text="tau"),y="Quantile Function")
  } else if(type == "PDF") {
    p <- p + labs(x="Y",y="Probability Density Function")
  } else {
    p <- p + labs(x="Y",y="Cumulative Density Function")
  }
  p +
    theme_bw() +
    theme(axis.text=element_text(colour="black", size = 12),
          axis.title=element_text(size=15))
}
