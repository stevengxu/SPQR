#' @title plot MCMC trace plots
#' @description
#' Show trace plot of the log-likelihood or estimates of a `SPQR` class object fitted using the MCMC method
#'
#' @import ggplot2
#' @export
mcmcTrace <- function(object, target = c("loglik","PDF","CDF","QF"),
                      X = NULL, Y = NULL, tau = 0.5, window = NULL) {
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
    loglik <- rowMeans(object$chain.info$loglik)
    data$target <- loglik[window[1]:window[2]]
    target <- "log-likelihood"
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
    labs(x="Post-warmup iteration", y=target) +
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
