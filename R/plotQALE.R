#' @title Plot accumulated local effects (ALE)
#' @description
#' Computes and plots the quantile ALEs of a `SPQR` class object. The function plots the ALE main effects across
#' \code{tau} for a single covariate using line plots, and the ALE interaction effects between two covariates
#' across \code{tau} using contour plots.
#'
#' @import ggplot2
#'
#' @param object An object of class \code{"SPQR"}.
#' @inheritDotParams QALE -object
#'
#' @return A \code{ggplot} object.
#'
#' @export
plotQALE <- function(object, ...) {

  dotparams <- list(...)
  var.index <- dotparams$var.index
  tau <- dotparams$tau
  ci.level <- dotparams$ci.level
  getAll <- dotparams$getAll
  if (is.null(var.index)) stop("`var.index` can not be NULL.")
  if (is.null(tau)) tau <- seq(0.1,0.9,0.1)
  if (is.null(ci.level)) ci.level <- 0
  if (is.null(getAll)) getAll <- FALSE

  N <- nrow(object$X)
  x.names <- colnames(object$X)
  if (!is.null(x.names)) x.label <- x.names[var.index]
  else x.label <- parse(text=paste0("X[",var.index,"]"))
  tauexp <- factor(tau, levels=tau, labels=paste0("tau==",tau))
  ale <- QALE(object, ...)
  yrange <- range(ale$ALE)
  yrange[2] <- max(yrange[2],0.1)
  yrange[1] <- min(yrange[1],-0.1)
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
        geom_line(aes(x=.data$x,y=.data$y,group=.data$g),alpha=0.1) +
        geom_line(data=dfm,aes(x=.data$x,y=.data$y),size=1,color="red") +
        facet_wrap(~tauexp, labeller=label_parsed) +
        theme(panel.spacing=unit(0, "lines"))
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
      fld <- with(xygrid, akima::interp(x=x, y=y, z=z))
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
