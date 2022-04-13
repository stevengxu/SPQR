#' @title plot quantile ALE-induced variable importance
#' @description
#'
#' @import ggplot2
#' @export
plotQVI <- function(object, var.index = NULL, var.names = NULL, tau = 0.5,
                    ci.level = 0, n.bins = 40) {
  if (is.null(var.index)) var.index <- 1:ncol(object$X)
  else stopifnot(length(var.index)>2)
  if (!is.null(var.names) && length(var.names) != length(var.index))
    stop("`var.names` should have the same length as 'var.index'.")

  if (object$method != "MCMC") ci.level <- 0
  x.ticks <- {}
  if (!is.null(var.names)) x.ticks <- var.names
  else x.ticks <- parse(text=paste0("X[",var.index,"]"))
  names(x.ticks) <- var.index
  tauexp <- factor(tau, levels=tau, labels=paste0("tau==",tau))
  if (ci.level == 0) {
    vi <- matrix(nrow=length(var.index),ncol=length(tau))
    for (i in 1:nrow(vi)) {
      ale <- QALE(object, var.index=var.index[i], n.bins=n.bins, tau=tau)
      if (length(ale$x)>5) vi[i,] <- apply(ale$ALE,2,stats::sd)
      else vi[i,] <- apply(ale$ALE,2,function(x) max(x)-min(x))/4
    }
    .df <- data.frame(x=paste0("X[",var.index,"]"))
    df <- do.call(rbind,lapply(seq_along(tau), FUN = function(i) {
      .df$y <- vi[,i]
      #x.ticks <- x.ticks[order(df$y, decreasing = T)]
      .df <- .df[order(.df$y, decreasing = T),]
      .df$tauexp <- tauexp[i]
      return(.df)
    }))
    p <-
      ggplot(data=df, aes(x=.reorder_within(.data$x,-.data$y,.data$tauexp), y=.data$y)) +
      geom_bar(stat="identity",fill="#999999")
  } else {
    nnn <- length(object$model)
    .vi <- array(dim=c(length(var.index),length(tau),nnn))
    vi <- array(dim=c(length(var.index),length(tau),3))
    for (i in 1:nrow(vi)) {
      ale <- QALE(object, var.index=var.index[i], n.bins=n.bins, tau=tau, getAll=TRUE)
      if (length(ale$x)>5) .vi[i,,] <- apply(ale$ALE,c(2,3),stats::sd)
      else .vi[i,,] <- apply(ale$ALE,c(2,3),function(x) max(x)-min(x))/4
      vi[i,,1] <- apply(.vi[i,,],1,stats::quantile,probs=(1-ci.level)/2)
      vi[i,,2] <- apply(.vi[i,,],1,mean)
      vi[i,,3] <- apply(.vi[i,,],1,stats::quantile,probs=(1+ci.level)/2)
    }
    .df <- data.frame(x=paste0("X[",var.index,"]"))
    df <- do.call(rbind,lapply(seq_along(tau), FUN = function(i) {
      .df$y <- as.vector(vi[,i,2])
      .df$ymin <- as.vector(vi[,i,1])
      .df$ymax <- as.vector(vi[,i,3])
      #x.ticks <- x.ticks[order(df$y, decreasing = T)]
      .df <- .df[order(.df$y, decreasing = T),]
      .df$tauexp <- tauexp[i]
      return(.df)
    }))
    p <-
      ggplot(data=df, aes(x=.reorder_within(.data$x,-.data$y,.data$tauexp), y=.data$y)) +
      geom_bar(stat="identity",fill="#999999") +
      geom_errorbar(aes(ymin=.data$ymin,ymax=.data$ymax),color="#000000")
  }
  p <- p +
    theme_bw() +
    scale_x_discrete(labels = function(x, sep = "___") {
      reg <- paste0(sep, ".+$")
      parse(text=gsub(reg, "", x))
    }) +
    facet_wrap(~tauexp, labeller=label_parsed, scales='free_x') +
    labs(x=NULL, y="Importance") +
    theme(panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          panel.spacing=unit(0, "lines"),
          axis.title=element_text(size = 15),
          axis.text=element_text(colour="black", size = 12),
          strip.text=element_text(size = 15))
  return(p)
}

.reorder_within <- function (x, by, within, fun = mean, sep = "___", ...) {
  if (!is.list(within)) within <- list(within)
  new_x <- do.call(paste, c(list(x, sep = sep), within))
  stats::reorder(new_x, by, FUN = fun)
}


