#' @title plot SPQR estimators
#' @description
#' Computes and plots the estimated PDF/CDF/QF curves.
#'
#' @name plotEstimator
#'
#' @param object An object of class \code{"SPQR"}
#' @param X A row vector indicating covariate values for which the conditional PDF/CDF/QF is computed and plotted.
#' @inheritDotParams predict.SPQR -X -Y
#'
#' @return A \code{ggplot} object.
#'
#' @import ggplot2
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
#'
#' ## plot estimated PDF
#' plotEstimator(fit, type = "PDF", X = 0)
#' }
#' @export
plotEstimator <- function(object, X, ...) {

  dotparams <- list(...)
  nY <- dotparams$nY
  type <- dotparams$type
  tau <- dotparams$tau
  ci.level <- dotparams$ci.level
  getAll <- dotparams$getAll
  if (is.null(nY)) nY <- 101
  if (is.null(type)) type <- "QF"
  if (is.null(tau)) tau <- seq(0.1,0.9,0.1)
  if (is.null(ci.level)) ci.level <- 0
  if (is.null(getAll)) getAll <- FALSE

  if (NROW(X) > 1 && NCOL(X) > 1)
    stop("`X` must contain exactly 1 observation")

  yy <- predict.SPQR(object, X=X, ...)
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
      geom_line(data=dfa, aes(x=.data$x,y=.data$y,group=.data$g), alpha=0.1) +
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
    p <- p + labs(x=parse(text="tau"),y="Quantile")
  } else if(type == "PDF") {
    p <- p + labs(x="Y",y="Density")
  } else {
    p <- p + labs(x="Y",y="Probability")
  }
  p +
    theme_bw() +
    theme(axis.text=element_text(colour="black", size = 12),
          axis.title=element_text(size=15))
}
