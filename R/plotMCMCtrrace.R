#' @title plot MCMC trace plots
#' @description
#' Show trace plot of the log-likelihood or estimates, of a \code{"SPQR"} class object fitted using the MCMC method
#'
#' @param object An object of class \code{SPQR}.
#' @param target A character indicating the statistic/estimate for which traceplot should be plotted;
#'   \code{"loglik"}: log-likelihood (default), \code{"PDF"}: probability density function, \code{"CDF"}: cumulative density function,
#'   \code{"QF"}: quantile function.
#' @param X If \code{target != "loglik"}, a row vector specifying the covariate values for which the estimates are computed. Default: \code{NULL}.
#' @param Y If \code{target == "PDF" || target == "CDF"} a scalar specifying the response value for which the estimates are computed. Default: \code{NULL}.
#' @param tau If \code{target != "QF"}, a scalar specifying the quantile level for which the estimates are computed. Default: 0.5.
#' @param window A vector specifying the range of index of the MCMC samples for which the traceplot should be plotted. Default is \code{NULL}
#'   indicating that the whole chain is plotted.
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
#' control <- list(iter = 300, warmup = 200, thin = 1)
#' fit <- SPQR(X = X, Y = Y, method = "MCMC", control = control, normalize = TRUE)
#'
#' ## traceplot for log-likelihood
#' plotMCMCtrace(fit, target = "loglik")
#'
#'
#' @export
plotMCMCtrace <- function(object, target = c("loglik","PDF","CDF","QF"),
                          X = NULL, Y = NULL, tau = 0.5, window = NULL) {
  if (object$method != "MCMC")
    stop("trace plot is only available for SPQR fitted with `method=\"MCMC\"`")
  target <- match.arg(target)
  divergent <- object$chain.info$divergent
  if (!is.null(window)) {
    stopifnot(length(window)==2)
    stopifnot(window[1]<window[2])
    stopifnot(window[1]>0)
    window[2] <- min(window[2], length(divergent))
  } else {
    window <- c(1,length(divergent))
  }
  divergent <- divergent[window[1]:window[2]]
  divergent <- ifelse(divergent, seq_along(divergent), NA)
  data <- data.frame(x=seq_along(divergent), divergent=divergent)
  if (sum(divergent, na.rm=T) > 0) data$divergent <- divergent
  if (target == "loglik") {
    loglik <- rowSums(object$chain.info$loglik)
    data$target <- loglik[window[1]:window[2]]
  } else {
    if (is.null(dim(X))) dim(X) <- c(1,length(X))
    stopifnot(NROW(X)==1)
    if (target == "PDF" || target == "CDF") {
      if (is.null(Y)) stop("`Y` cannot be NULL")
      stopifnot(NROW(Y)==1)
    }
    if (target == "QF") stopifnot(length(tau)==1)
    data$target <-
      predict.SPQR(object=object, X=X, Y=Y, type=target, tau=tau, getAll=TRUE)[window[1]:window[2]]
  }
  ylab <- switch(target,
                 `QF` = "Quantile",
                 `PDF` = "Density",
                 `CDF` = "Probability",
                 `loglik` = "log-Likelihood")
  p <-
    ggplot(data=data) +
    geom_line(aes(x=.data$x,y=.data$target),color="#414487FF") +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.title = element_text(size = 15),
          plot.title = element_text(hjust = 0.5, size = 18),
          axis.text.y = element_text(size = 12),
          axis.text.x = element_text(size = 12)) +
    labs(x="Post-warmup iteration", y=ylab) +
    scale_x_continuous(breaks = pretty)

  if (sum(divergent, na.rm=T)>0) {
    p <- p +
      geom_rug(aes(x=.data$divergent, color="Divergence"),
               na.rm = TRUE,
               sides = "b") +
      scale_color_manual(values = "red", name = NULL) +
      theme(legend.text = element_text(size = 12))
  }
  return(p)
}
