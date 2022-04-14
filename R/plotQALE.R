#' @title Plot accumulated local effects (ALE)
#' @description
#' Computes and plots the quantile ALEs of a `SPQR` class object. The function plots the ALE main effects across
#' \code{tau} for a single covariate using line plots, and the ALE interaction effects between two covariates
#' across \code{tau} using contour plots.
#'
#' @import ggplot2
#'
#' @param object An object of class \code{"SPQR"}.
#' @param var.index a numeric scalar or length-two vector of indices of the covariates for
#'   which the ALEs will be calculated. When \code{length(var.index)==1},
#'   the function computes the main effect for \code{X[,var.index]}. When \code{length(var.index)==2},
#'   the function computes the interaction effect between \code{X[,var.index[1]]} and \code{X[,var.index[2]]}.
#' @param tau The quantiles of interes.
#' @param n.bins  the maximum number of intervals into which the covariate range is divided when
#'   calculating the ALEs. The actual number of intervals depends on the number of unique values in
#' \code{X[,var.index]}. When \code{length(var.index)==2}, \code{n.bins} is applied to both covariates.
#' @param ci.level The credible level for plotting the credible bands for ALE
#'   when \code{length(var.index)==1}. The default is 0 indicating no credible bands should be plotted.
#' @param getAll If \code{TRUE} and \code{length(var.index)==1}, plots all posterior samples of ALE.
#'
#' @return A \code{ggplot} object.
#'
#' @export
plotQALE <- function(object, var.index = NULL, tau = 0.5, ci.level = 0,
                     getAll = FALSE, n.bins = 40) {
  if (length(var.index) > 2)
    stop("For ALE plot, `var.index` should be a vector of length one or two.")
  if (object$method != "MCMC") {
    ci.level <- 0
    getAll <- FALSE
  }
  N <- nrow(object$X)
  x.names <- colnames(object$X)
  if (!is.null(x.names)) x.label <- x.names[var.index]
  else x.label <- parse(text=paste0("X[",var.index,"]"))
  tauexp <- factor(tau, levels=tau, labels=paste0("tau==",tau))
  ale <- QALE(object, var.index=var.index, n.bins=n.bins, tau=tau, ci.level=ci.level, getAll=getAll)
  yrange <- range(ale$ALE)
  yrange[2] <- max(yrange[2],0.1)
  yrange[1] <- min(yrange[1],-yrange[2])
  if (length(var.index) == 1) {
    K <- length(ale$x)
    if (getAll) {
      nnn <- length(object$model)
      ym <- apply(ale$ALE,1:2,mean)
      dfm <- data.frame(x=ale$x,
                        y=as.vector(ym),
                        tau=rep(tau,each=K),
                        tauexp=rep(tauexp,each=K))
      dfa <- data.frame(x=ale$x,
                        y=as.vector(ale$ALE),
                        g=rep(1:nnn,each=K*length(tau)),
                        tau=rep(tau,each=K),
                        tauexp=rep(tauexp,each=K))
      p <-
        ggplot(data=dfa) +
        theme_bw() +
        geom_line(aes(x=.data$x,y=.data$y,group=.data$g),alpha=0.2) +
        geom_line(data=dfm,aes(x=.data$x,y=.data$y),size=1,color="red") +
        facet_wrap(~tauexp, labeller=label_parsed) +
        theme(panel.spacing = unit(0, "lines"))
    } else if (ci.level > 0){
      df <- data.frame(x=ale$x,
                       y=as.vector(ale$ALE[,,2]),
                       ymin=as.vector(ale$ALE[,,1]),
                       ymax=as.vector(ale$ALE[,,3]),
                       tau=rep(tau,each=K),
                       tauexp=rep(tauexp,each=K))
      p <-
        ggplot(data=df) +
        theme_bw() +
        geom_line(aes(x=.data$x,y=.data$y),size=1) +
        geom_ribbon(aes(x=.data$x,ymin=.data$ymin,ymax=.data$ymax),alpha=0.3) +
        facet_wrap(~tauexp, labeller=label_parsed) +
        theme(panel.spacing = unit(0, "lines"))
    } else {
      df <- data.frame(x=ale$x,y=as.vector(ale$ALE),tau=rep(tau,each=K),
                       tauexp=rep(tauexp,each=K))
      p <-
        ggplot(data=df) +
        theme_bw() +
        geom_line(aes(x=.data$x,y=.data$y),size=1) +
        facet_wrap(~tauexp, labeller=label_parsed) +
        theme(panel.spacing = unit(0, "lines"))
    }
    p <- p +
      labs(x=x.label, y="ALE") +
      ylim(yrange)
  } else {
    xygrid <- expand.grid(x=ale$x[[1]],y=ale$x[[2]])
    df <- do.call(rbind,lapply(seq_along(tau),function(i){
      xygrid$z <- as.vector(ale$ALE[,,i])
      fld <- with(xygrid, interp(x=x, y=y, z=z))
      out <- expand.grid(x=fld$x, y=fld$y)
      out$z <- c(fld$z)
      out$tauexp <- rep(tauexp[i],nrow(out))
      return(out)
    }))
    p <- ggplot(data=df, aes(x=.data$x,y=.data$y)) +
      theme_bw() +
      geom_raster(aes(fill=.data$z)) +
      geom_contour(aes(z=.data$z),colour="black") +
      scale_fill_gradientn(colors=rev(RColorBrewer::brewer.pal(10,"Spectral")),
                           name=parse(text="ALE"), limits=yrange) +
      facet_wrap(~tauexp, labeller=label_parsed) +
      labs(x=x.label[1], y=x.label[2]) +
      theme(panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            panel.spacing=unit(0, "lines"))
  }
  p +
    theme(axis.text=element_text(colour="black", size = 12),
          axis.title=element_text(size=15),
          strip.text=element_text(size=15),
          legend.title=element_text(size=18),
          legend.text=element_text(size=12))
}
