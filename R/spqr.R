library(rlang)
library(Rcpp)
library(loo)
library(rstan)
library(progressr)
library(matrixStats)
library(torch)
library(splines2)
library(yaImpute)
library(abind)
library(ggplot2)
library(gridExtra)
library(RColorBrewer)
library(metR)
library(akima)
source("utils.R")
source("SPQR.mcmc.R")
source("SPQR.adam.R")


SPQR <- function(params = list(), X, Y, normalize = FALSE, verbose = TRUE, ...) {
  
  if (is.null(n <- nrow(X))) dim(X) <- c(length(X),1) # 1D matrix case
  if (n == 0) stop("`X` is empty")
  if (!is.matrix(X)) X <- as.matrix(X) # data.frame case
  
  ny <- NCOL(Y)
  if (is.matrix(Y) && ny == 1) Y <- drop(Y) # treat 1D matrix as vector
  if (NROW(Y) != n) stop("incompatible dimensions")
  
  # normalize all covariates
  if (normalize) {
    Y.range <- range(Y)
    Y <- (Y - Y.range[1])/diff(Y.range)
    X.range <- apply(X,2,range)
    X <- apply(X,2,function(x){
      (x - min(x)) / (max(x) - min(x))
    })
  }
  if (min(Y)<0 || max(Y)>1) stop("values of `Y` should be between 0 and 1")
  
  params <- check.SPQR.params(params, ...)
  if (params$method == "Bayes") {
    out <- SPQR.mcmc(params=params, X=X, Y=Y, verbose=verbose)
  } else {
    out <- SPQR.adam(params=params, X=X, Y=Y, verbose=verbose)
  }
  if (normalize) {
    out$normalize$X <- X.range
    out$normalize$Y <- Y.range
  }
  class(out) <- "SPQR"
  invisible(out)
}

print.SPQR <- function(object, showModel = FALSE) {
  s <- summary.SPQR(object)
  print.summary.SPQR(s, showModel = showModel)
}

summary.SPQR <- function(object) {
  
  method <- object$params$method
  out <- list(method=method, time=object$time)
  if (method != "MLE") out$prior <- object$params$prior
  out$model <- list(n.inputs=ncol(object$X),
                    n.knots=object$params$n.knots,
                    n.hidden=object$params$n.hidden,
                    activation=object$params$activation)
  if (method == "Bayes") {
    ll.mat <- object$chain.info$loglik
    ll.vec <- rowMeans(ll.mat)
    suppressWarnings(waic <- loo::waic(ll.mat)$estimates[1]) # Calculate WAIC
    reff <- relative_eff(exp(ll.mat), chain_id=rep(1,nrow(ll.mat)))
    suppressWarnings(loo <- loo::loo(ll.mat, r_eff=reff)$estimates[1]) # Calculate LOOIC
    bess <- rstan::ess_bulk(as.matrix(ll.vec))
    tess <- rstan::ess_tail(as.matrix(ll.vec))
    ndiv <- sum(object$chain.info$divergent)
    out$elpd <- list(loo=loo, waic=waic)
    accept.ratio <- mean(object$chain.info$accept.ratio)
    delta <- object$chain.info$delta
    out$diagnostics <- list(bulk.ess=bess,tail.ess=tess,ndiv=ndiv,
                            accept.ratio=accept.ratio, delta=delta)
  } else {
    out$loss <- object$loss
    out$optim.info <- list(lr=object$params$lr,
                           batch.size=object$params$batch.size)
  }
  class(out) <- "summary.SPQR"
  return(out)
}

print.summary.SPQR <- function(object, showModel = FALSE) {
  s <- object
  method <- s$method
  if (method != "MLE")
    cat("\nSPQR fitted using ", method, " approach with ", s$prior, " prior", sep="")
  else
    cat("\nSPQR fitted using ", method, " approach", sep="")
  cat("\U0001f680\n")
  
  if (method != "Bayes") {
    lr <- object$optim.info$lr
    batch.size <- object$optim.info$batch.size
    cat("\nLearning rate: ", lr, sep="")
    cat("\nBatch size: ", batch.size, "\n", sep="")
  }
    
  
  
  if (showModel) {
    cat("\nModel specification:\n")
    cat("  ")
    .printNNmat(object$model)
  }
    
  if (method == "Bayes") {
    
    bess <- s$diagnostics$bulk.ess
    tess <- s$diagnostics$tail.ess
    ndiv <- s$diagnostics$ndiv
    loo <- s$elpd$loo
    waic <- s$elpd$waic
    accept.ratio <- s$diagnostics$accept.ratio
    delta <- s$diagnostics$delta
    
    cat("\nMCMC diagnostics:\n",
        "  bulk.ESS = ", bess, ",  tail.ESS = ", tess, "\n", sep="")
    cat("  Final acceptance ratio is ", sprintf("%.2f", accept.ratio), " and target is ", delta, "\n", sep="") 
    if (s$diagnostics$ndiv > 0)
      cat("  There were ", paste0(ndiv, " divergent transitions after warmup"), "\n", sep="")
    
    cat("\nExpected log pointwise predictive density (elpd) estimates:\n", 
        "  elpd.LOO = ", loo, ",  elpd.WAIC = ", waic, "\n", sep="")
  } else {
    tr <- s$loss$train
    va <- s$loss$validation
    cat("\nLoss:\n",
        "  train = ", tr, ",  validation = ", va, "\n", sep="")
  }
  cat("\nElapsed time: ", paste0(sprintf("%.1f", s$time), " minutes"), "\n", sep = "")
}

.printNNmat <- function(model) {
  n.layers <- length(model$n.hidden) + 1
  nodes <- c(model$n.inputs, model$n.hidden, model$n.knots)
  mat <- array("", dim=c(n.layers,3), 
               dimnames=list(" "=rep("",n.layers),"Layers"=c("Input","Output","Activation")))
  for (l in 1:n.layers) mat[l,] <- c(nodes[l],nodes[l+1],model$activation)
  print.default(mat, quote = FALSE, right = TRUE)
} 

coef.SPQR <- function(object, X) {
  
  p <- ncol(object$X)
  if (NCOL(X) != p) {
    if (NROW(X) != p) stop("incompatible dimensions")
    else dim(X) <- c(1,length(X)) # treat vector as single observation
  }
  if (!is.null(object$normalize)) {
    X.range <- object$normalize$X
    for (j in 1:p) {
      X[,p] <- (X[,p] - X.range[1,p])/(diff(X.range[,p]))
    }
  }
  if (object$params$method == "Bayes") {
    X <- t(X)
    n <- ncol(X)
    nnn <- length(object$model)
    out <- rowMeans(sapply(object$model, function(W){
      .coefs(W,X,object$params$activation)
    }))
    dim(out) <- c(n, object$params$n.knots)
  } else {
    model <- object$model
    model$eval()
    out <- as.matrix(model(torch_tensor(X))$output)
  }
  colnames(out) <- paste0("theta[",1:object$params$n.knots,"]")
  names(dimnames(out)) <- c("X","Coefs")
  return(out)
}

qqCheck <- function(object, ci.level = 0, getAll = FALSE) {
  
  stopifnot(is.numeric(ci.level))
  if (ci.level < 0 || ci.level >=1) stop("`ci.level` should be between 0 and 1")
  
  B <- .basis(Y, K = object$params$n.knots, integral = TRUE)
  qu <- ppoints(max(1e3,length(object$Y)))
  if (object$params$method == "Bayes") {
    X <- t(object$X)
    n <- ncol(X)
    if (getAll) {
      cdf <- sapply(object$model,function(W){
        coefs <- .coefs(W, X, object$params$activation)
        rowSums(coefs*B)})
      .qqplot(qu,cdf,getAll=TRUE)
    } else if (ci.level > 0) {
      .cdf <- sapply(object$model,function(W){
        coefs <- .coefs(W, X, object$params$activation)
        rowSums(coefs*B)})
      cdf <- matrix(nrow=nrow(.cdf),ncol=3)
      cdf[,1] <- apply(.cdf,1,quantile,probs=(1-ci.level)/2)
      cdf[,2] <- apply(.cdf,1,mean)
      cdf[,3] <- apply(.cdf,1,quantile,probs=(1+ci.level)/2)
      .qqplot(qu,cdf,ci.level=ci.level)
    } else {
      coefs <- rowMeans(sapply(object$model, function(W){
        .coefs(W,X,object$params$activation)
      }))
      dim(coefs) <- c(n, object$params$n.knots)
      cdf <- rowSums(coefs*B)
      .qqplot(qu,cdf)
    }
  } else {
    model <- object$model
    model$eval()
    coefs <- as.matrix(model(torch_tensor(X))$output)
    cdf <- rowSums(coefs*B)
    .qqplot(qu,cdf)
  }
}

.qqplot <- function(x, y, ci.level = 0, getAll = FALSE){
  
  lenx <- length(x)
  sx <- sort(x)
  leny <- NROW(y)
  if (leny < lenx) sx <- approx(1L:lenx, sx, n = leny)$y
  if (getAll) {
    sy <- apply(y,2,sort)
    my <- sort(rowMeans(y))
    dat <- data.frame(sx=sx, my=my)
    data <- data.frame(sx=sx, sy=c(sy), g=rep(1:ncol(sy),each=nrow(sy)))
    p <-
      ggplot() + 
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_point(data=dat, aes(x=sx, y=my), alpha=0.3, shape=19, size=2) +
      geom_line(data=data, aes(x=sx, y=sy, group=g), alpha=0.2)
  } else if (ci.level > 0){
    oy <- order(y[,2])
    sy <- y[oy,]
    data <- data.frame(sx = sx, sy = sy[,2], ymin = sy[,1], ymax = sy[,3])
    p <- 
      ggplot(data, aes(x=sx, y=sy)) + 
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_ribbon(aes(x=sx, ymin=ymin, ymax=ymax), alpha=0.3) +
      geom_point(alpha=0.3, shape=19, size=2)
  } else if (is.numeric(ci.level)) {
    sy <- sort(y)
    data <- data.frame(sx = sx, sy = sy)
    p <- 
      ggplot(data, aes(x=sx, y=sy)) + 
      geom_abline(intercept=0, slope=1, color="red",size=1.5, linetype=2) +
      geom_point(alpha=0.3, shape=19, size=2) 
  }
  p + theme_bw() + labs(title="Q-Q Plot", x="Unif(0,1)", y="Fitted CDF") +
    theme(axis.text = element_text(colour="black", size = 12),
          axis.title=element_text(size=15),
          plot.title=element_text(hjust=0.5, size=18))
}

predict.SPQR <- function(object, X, Y = NULL, nY = 501, type = c("QF","PDF","CDF"), 
                         tau = seq(0.05,0.95,0.05), ci.level = 0, getAll = FALSE) {
  type <- match.arg(type)
  stopifnot(is.numeric(ci.level))
  if (ci.level < 0 || ci.level >=1) stop("`ci.level` should be between 0 and 1")
  
  Y.normalize <- X.normalize <- !is.null(object$normalize)
  p <- ncol(object$X)
  if (NCOL(X) != p) {
    if (NROW(X) != p) stop("incompatible dimensions")
    else dim(X) <- c(1,length(X)) # treat vector as single observation
  }
  if (X.normalize) {
    X.range <- object$normalize$X
    for (j in 1:p) {
      X[,p] <- (X[,p] - X.range[1,p])/(diff(X.range[,p]))
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
  
  B <- .basis(Y, K = object$params$n.knots, integral = (type != "PDF"))
  if (object$params$method == "Bayes") {
    X <- t(X)
    n <- ncol(X)
    nest <- if (type == "QF") length(tau) else length(Y) 
    nnn <- length(object$model)
    if (getAll) {
      out <- array(dim=c(n,nest,nnn))
      for (i in 1:nnn) {
        coefs <- .coefs(object$model[[i]], X, object$params$activation)
        out[,,i] <- .predict.SPQR(Y,coefs,B,type,tau)
      }
      dimnames(out)[[3]] <- 1:nnn
      names(dimnames(out))[3] <- "Iteration"
    } else if (ci.level > 0) {
      .out <- array(dim=c(n,nest,nnn))
      for (i in 1:nnn) {
        coefs <- .coefs(object$model[[i]], X, object$params$activation)
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
        coefs <- coefs + .coefs(object$model[[i]],X,object$params$activation)/nnn
      }
      out <- .predict.SPQR(Y, coefs, B, type, tau)
    }
  } else {
    model <- object$model
    model$eval()
    coefs <- as.matrix(model(torch_tensor(X))$output)
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
  for (ii in 1:nrow(qf)) qf[ii,] <- approx(df[ii,], Y, xout=tau, ties = list("ordered", min))$y 
  return(qf)
}

cv.SPQR <- function(params = list(), X, Y, nfold, folds=NULL, stratified=FALSE, 
                    verbose = TRUE, ...) {
  params <- check.SPQR.params(params, ...)
  if (params$method == "Bayes") {
    stop("Cross-validation not available for `method` = 'Bayes'") 
  }
  if (!is.null(folds)) {
    if (!is.list(folds) || length(folds) < 2)
      stop("`folds` must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    nfold <- length(folds)
  } else {
    if (nfold <= 1)
      stop("`nfold` must be > 1")
    folds <- SPQR.createFolds(Y, nfold, stratified)
  }
  out <- cv.SPQR.adam(params=params, X=X, Y=Y, folds=folds, verbose=verbose)
  invisible(out)
}


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
    geom_line(aes(x=x,y=target),color="#414487FF") +
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
      geom_rug(aes(x=divergent, color="Divergence"),
               na.rm = TRUE,
               sides = "b") +
      scale_color_manual(values = "red", name = NULL) + 
      theme(legend.text = element_text(size = 12))
  }
  return(p)
}

plot.SPQR <- function(object, X, nY=501, type = c("QF","PDF","CDF"),
                      tau = seq(0.05,0.95,0.05), ci.level = 0, getAll = FALSE) {
  type <- match.arg(type)
  if (NROW(X) > 1 && NCOL(X) > 1)
    stop("`X` should contain exactly 1 observation")
  if (object$params$method != "Bayes") {
    ci.level <- 0
    getAll <- FALSE
  }
  yy <- predict(object, X=X, nY=nY, type=type, tau=tau, ci.level=ci.level, getAll=getAll)
  if (type != "QF") {
    # need Y on original scale for plotting
    Y <- seq(0,1,length.out=nY)
    Y.normalize <- !is.null(object$normalize)
    if (Y.normalize) {
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
      geom_line(data=dfa, aes(x=x,y=y,group=g), alpha=0.2) +
      geom_line(data=dfm, aes(x=x,y=y), size=1, color="red")
  } else if (ci.level > 0) {
    df <- data.frame(x=xx, y=yy[,,2], ymin=yy[,,1], ymax=yy[,,3])
    p <- 
      ggplot(data=df, aes(x=x, y=y)) + 
      geom_line(size=1) +
      geom_ribbon(aes(x=x, ymin=ymin, ymax=ymax), alpha=0.2)
  } else {
    df <- data.frame(x=xx, y=yy)
    p <- 
      ggplot(data=df, aes(x=x,y=y)) + 
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

plotQALE <- function(object, var.index = NULL, tau = 0.5, ci.level = 0, 
                     getAll = FALSE, singlePlot = FALSE, n.bins = 40) {
  if (length(var.index) > 2)
    stop("For ALE plot, `var.index` should be a vector of length one or two.")
  if (object$params$method != "Bayes") {
    ci.level <- 0
    getAll <- FALSE
  }
  if (ci.level || getAll || length(var.index)==2) singlePlot <- FALSE
  N <- nrow(object$X)
  x.names <- colnames(X)
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
      p <- ggplot(data=dfa) + theme_bw()
      if (singlePlot) {
        p <- p + 
          geom_line(aes(x=x,y=y,group=tau),alpha=0.2) +
          geom_line(data=dfm,aes(x=x,y=y,color=factor(tau)),group=tau) + 
          guides(color=guide_legend(title=parse(text="tau")))
      } else {
        p <- p + 
          geom_line(aes(x=x,y=y,group=g),alpha=0.2) + 
          geom_line(data=dfm,aes(x=x,y=y),size=1,color="red") +
          facet_wrap(~tauexp, labeller=label_parsed) +
          theme(panel.spacing = unit(0, "lines"))
      }
    } else if (ci.level > 0){
      df <- data.frame(x=ale$x,
                       y=as.vector(ale$ALE[,,2]),
                       ymin=as.vector(ale$ALE[,,1]),
                       ymax=as.vector(ale$ALE[,,3]),
                       tau=rep(tau,each=K),
                       tauexp=rep(tauexp,each=K))
      p <- ggplot(data=df) + theme_bw()
      if (singlePlot) {
        p <- p + 
          geom_line(aes(x=x,y=y,group=tau,color=factor(tau)),size=1) +
          geom_ribbon(aes(x=x,ymin=ymin,ymax=ymax,group=tau),alpha=0.3)  +  
          guides(color=guide_legend(title=parse(text="tau")))
      } else {
        p <- p + 
          geom_line(aes(x=x,y=y),size=1) + 
          geom_ribbon(aes(x=x,ymin=ymin,ymax=ymax),alpha=0.3) +
          facet_wrap(~tauexp, labeller=label_parsed) + 
          theme(panel.spacing = unit(0, "lines"))
        
      }
    } else {
      df <- data.frame(x=ale$x,y=as.vector(ale$ALE),tau=rep(tau,each=K),
                       tauexp=rep(tauexp,each=K))
      p <- ggplot(data=df) + theme_bw()
      if (singlePlot) {
        p <- p + 
          geom_line(aes(x=x,y=y,group=tau,color=factor(tau)),size=1) + 
          guides(color=guide_legend(title=parse(text="tau")))
      } else {
        p <- p + 
          geom_line(aes(x=x,y=y),size=1) + 
          facet_wrap(~tauexp, labeller=label_parsed) + 
          theme(panel.spacing = unit(0, "lines"))
          
      }
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
    p <- ggplot(data=df, aes(x=x,y=y)) + 
      theme_bw() +
      geom_raster(aes(fill=z)) + 
      geom_contour(aes(z=z),colour="black") +
      scale_fill_gradientn(colors=rev(brewer.pal(10,"Spectral")),
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

plotQVI <- function(object, var.index = NULL, var.names = NULL, tau = 0.5, 
                    ci.level = 0, n.bins = 40) {
  if (is.null(var.index)) var.index <- 1:ncol(object$X)
  else stopifnot(length(var.index)>2)
  if (!is.null(var.names) && length(var.names) != length(var.index))
    stop("`var.names` should have the same length as 'var.index'.")
  
  if (object$params$method != "Bayes") ci.level <- 0
  x.ticks <- {}
  if (!is.null(var.names)) x.ticks <- var.names
  else x.ticks <- parse(text=paste0("X[",var.index,"]"))
  names(x.ticks) <- var.index
  tauexp <- factor(tau, levels=tau, labels=paste0("tau==",tau))
  if (ci.level == 0) {
    vi <- matrix(nrow=length(var.index),ncol=length(tau))
    for (i in 1:nrow(vi)) {
      ale <- QALE(object, var.index=var.index[i], n.bins=n.bins, tau=tau)
      if (length(ale$x)>5) vi[i,] <- apply(ale$ALE,2,sd)
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
      ggplot(data=df, aes(x=.reorder_within(x,-y,tauexp), y=y)) +
      geom_bar(stat="identity",fill="#999999")
  } else {
    nnn <- length(object$model)
    .vi <- array(dim=c(length(var.index),length(tau),nnn))
    vi <- array(dim=c(length(var.index),length(tau),3))
    for (i in 1:nrow(vi)) {
      ale <- QALE(object, var.index=var.index[i], n.bins=n.bins, tau=tau, getAll=TRUE)
      if (length(ale$x)>5) .vi[i,,] <- apply(ale$ALE,c(2,3),sd)
      else .vi[i,,] <- apply(ale$ALE,c(2,3),function(x) max(x)-min(x))/4
      vi[i,,1] <- apply(.vi[i,,],1,quantile,probs=(1-ci.level)/2)
      vi[i,,2] <- apply(.vi[i,,],1,mean)
      vi[i,,3] <- apply(.vi[i,,],1,quantile,probs=(1+ci.level)/2)
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
      ggplot(data=df, aes(x=.reorder_within(x,-y,tauexp), y=y)) +
      geom_bar(stat="identity",fill="#999999") +
      geom_errorbar(aes(ymin=ymin,ymax=ymax),color="#000000")
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

QALE <- function(object, var.index, tau, n.bins = 40, ci.level = 0, 
                 getAll = FALSE, pred.fun = NULL) {
  
  if (!is.null(pred.fun) || object$params$method != "Bayes") {
    ci.level <- 0
    getAll <- FALSE
  }
  stopifnot(length(var.index) <= 2)
  stopifnot(is.numeric(ci.level))
  if (ci.level < 0 || ci.level >=1) stop("`ci.level` should be between 0 and 1")
  if (ci.level > 0) getAll <- TRUE
  if (!is.null(pred.fun)) stopifnot(is.function(pred.fun))
  
  X <- object$X
  N <- nrow(X)  # sample size
  d <- ncol(X)  # number of predictor variables
  J <- var.index # predictor index
  K <- n.bins # number of partition on each X_j
  
  firstCheck <- class(X[,J[1]]) == "numeric" || class(X[,J[1]]) == "integer"
  if (length(J) == 1) { # calculate main effects ALE plot
    if (!firstCheck)
      stop("X[,var.index] must be numeric or integer.")
    # find the vector of z values corresponding to the quantiles of X[,J]
    z <- c(min(X[,J]), as.numeric(quantile(X[,J],seq(1/K,1,length.out=K), type=1)))  # vector of K+1 z values
    z <- unique(z)  # necessary if X[,J] is discrete, in which case z could have repeated values 
    K <- length(z)-1 # reset K to the number of unique quantile points
    fJ <- numeric(K)
    # group training rows into bins based on z
    a1 <- as.numeric(cut(X[,J], breaks=z, include.lowest=TRUE)) # N-length index vector indicating into which z-bin the training rows fall
    X1 <- X
    X2 <- X
    X1[,J] <- z[a1]
    X2[,J] <- z[a1+1]
    if (is.null(pred.fun)) {
      y.hat1 <- predict.SPQR(object=object, X=X1, type="QF", tau=tau, getAll=getAll)
      y.hat2 <- predict.SPQR(object=object, X=X2, type="QF", tau=tau, getAll=getAll)
    } else {
      y.hat1 <- pred.fun(X=X1, tau=tau)
      y.hat2 <- pred.fun(X=X2, tau=tau)
    }
    Delta <- y.hat2-y.hat1 
    if (is.null(dim(Delta))) dim(Delta) <- c(N,1)
    if (getAll) {
      nnn <- length(object$model)
      # Delta is N x length(tau) x nnn
      DDelta <- array(0, dim = c(K, length(tau), nnn))
      fJ <- array(0, dim=c(K+1, length(tau), nnn))
      for (i in 1:nnn) {
        for (j in 1:length(tau)) {
          DDelta[,j,i] <- as.numeric(tapply(Delta[,j,i], a1, mean))
        }
        fJ[,,i] <- rbind(0,apply(DDelta[,,i,drop=FALSE],2,cumsum))
      }
      if (ci.level > 0) {
        .fJ <- array(0,dim=c(K+1, length(tau), 3))
        .fJ[,,1] <- apply(fJ,1:2,quantile,probs=(1-ci.level)/2)
        .fJ[,,2] <- apply(fJ,1:2,mean)
        .fJ[,,3] <- apply(fJ,1:2,quantile,probs=(1+ci.level)/2)
        fJ <- .fJ
        rm(.fJ)
        dimnames(fJ)[[3]] <- c("lower.bound","mean","upper.bound")
        names(dimnames(fJ))[3] <- "CI"
      } else {
        dimnames(fJ)[[3]] <- 1:nnn
        names(dimnames(fJ))[3] <- "Iteration"
      }
    } else {
      # Delta is N x length(tau)
      DDelta <- matrix(0, nrow = K, ncol = length(tau))
      for (i in 1:length(tau)) {
        DDelta[,i] <- as.numeric(tapply(Delta[,i], a1, mean)) #K-length vector of averaged local effect values
      }
      fJ <- rbind(0,apply(DDelta,2,cumsum)) #K+1 length vector
    }
    x <- z
    colnames(fJ) <- paste0(tau*100, "%")
    names(dimnames(fJ))[1:2] <- c("X","tau")
  #end of if (length(J) == 1) statement  
  } else { #calculate second-order effects ALE plot
    secondCheck <- class(X[,J[2]]) == "numeric" || class(X[,J[2]]) == "integer"
    if (!(firstCheck && secondCheck))
      stop("Both X[,var.index[1]] and X[,var.index[2]] must be numeric or integer.")
    
    #find the vectors of z values corresponding to the quantiles of X[,J[1]] and X[,J[2]]
    z1 <- c(min(X[,J[1]]), as.numeric(quantile(X[,J[1]],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values for X[,J[1]]
    z1 <- unique(z1)  #necessary if X[,J(1)] is discrete, in which case z1 could have repeated values 
    K1 <- length(z1)-1 #reset K1 to the number of unique quantile points
    if (K1 == 1)
      stop("X[,var.index[1]] should have at least 3 unique values.")
    #group training rows into bins based on z1
    a1 <- as.numeric(cut(X[,J[1]], breaks=z1, include.lowest=TRUE)) #N-length index vector indicating into which z1-bin the training rows fall
    z2 <- c(min(X[,J[2]]), as.numeric(quantile(X[,J[2]],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values for X[,J[2]]
    z2 <- unique(z2)  #necessary if X[,J(2)] is discrete, in which case z2 could have repeated values 
    K2 <- length(z2)-1 #reset K2 to the number of unique quantile points
    if (K2 == 1)
      stop("X[,var.index[2]] should have at least 3 unique values.")
    fJ <- matrix(0,K1,K2)  #rows correspond to X[,J(1)] and columns to X[,J(2)]
    #group training rows into bins based on z2
    a2 <- as.numeric(cut(X[,J[2]], breaks=z2, include.lowest=TRUE)) #N-length index vector indicating into which z2-bin the training rows fall
    X11 <- X  #matrix with low X[,J[1]] and low X[,J[2]]
    X12 <- X  #matrix with low X[,J[1]] and high X[,J[2]]
    X21 <- X  #matrix with high X[,J[1]] and low X[,J[2]]
    X22 <- X  #matrix with high X[,J[1]] and high X[,J[2]]
    X11[,J] <- cbind(z1[a1], z2[a2])
    X12[,J] <- cbind(z1[a1], z2[a2+1])
    X21[,J] <- cbind(z1[a1+1], z2[a2])
    X22[,J] <- cbind(z1[a1+1], z2[a2+1])
    if (is.null(pred.fun)) {
      y.hat11 <- predict.SPQR(object=object, X=X11, type="QF", tau=tau)
      y.hat12 <- predict.SPQR(object=object, X=X12, type="QF", tau=tau)
      y.hat21 <- predict.SPQR(object=object, X=X21, type="QF", tau=tau)
      y.hat22 <- predict.SPQR(object=object, X=X22, type="QF", tau=tau)
    } else {
      y.hat11 <- pred.fun(X=X11, tau=tau)
      y.hat12 <- pred.fun(X=X12, tau=tau)
      y.hat21 <- pred.fun(X=X21, tau=tau)
      y.hat22 <- pred.fun(X=X22, tau=tau)
    }
    .Delta <- (y.hat22-y.hat21)-(y.hat12-y.hat11)  #N-length vector of individual local effect values
    if (is.null(dim(.Delta))) dim(.Delta) <- c(N, length(tau))
    Delta <- array(dim=c(K1,K2,length(tau)))
    for (dd in 1:length(tau)) {
      #K1xK2 matrix of averaged local effects, which includes NA values if a cell is empty
      Delta[,,dd] <- as.matrix(tapply(.Delta[,dd], list(a1, a2), mean))
    }
    #replace NA values in Delta by the Delta value in their nearest neighbor non-NA cell
    NA.Delta <- is.na(Delta[,,1])  #K1xK2 matrix indicating cells that contain no observations
    NA.ind <- which(NA.Delta, arr.ind=T, useNames = F)  #2-column matrix of row and column indices for NA cells
    if (!is.null(nrow(NA.ind))) {
      if (nrow(NA.ind) > 0) {
        notNA.ind <- which(!NA.Delta, arr.ind=T, useNames = F)  #2-column matrix of row and column indices for non-NA cells
        range1 <- max(z1)-min(z1) 
        range2 <- max(z2)-min(z2)
        Z.NA <- cbind((z1[NA.ind[,1]] + z1[NA.ind[,1]+1])/2/range1, (z2[NA.ind[,2]] + z2[NA.ind[,2]+1])/2/range2) #standardized {z1,z2} values for NA cells corresponding to each row of NA.ind
        Z.notNA <- cbind((z1[notNA.ind[,1]] + z1[notNA.ind[,1]+1])/2/range1, (z2[notNA.ind[,2]] + z2[notNA.ind[,2]+1])/2/range2) #standardized {z1,z2} values for non-NA cells corresponding to each row of notNA.ind
        nbrs <- ann(Z.notNA, Z.NA, k=1, verbose = F)$knnIndexDist[,1] #vector of row indices (into Z.notNA) of nearest neighbor non-NA cells for each NA cell
        for (dd in 1:length(tau)) {
          Delta[,,dd][NA.ind] <- Delta[,,dd][matrix(notNA.ind[nbrs,], ncol=2)]
        }#Set Delta for NA cells equal to Delta for their closest neighbor non-NA cell. 
      } #end of if (nrow(NA.ind) > 0) statement
      #accumulate the values in Delta
    }
    fJ <- array(0,c(K1,K2,length(tau)))
    for (dd in 1:length(tau)) {
      fJ[,,dd] <- apply(t(apply(Delta[,,dd],1,cumsum)),2,cumsum)  
    }
    .fJ <- fJ
    fJ <- array(0,c(K1+1,K2+1,length(tau)))
    fJ[-1,-1,] <- .fJ
    #add a first row and first column to fJ that are all zeros
    #now subtract the lower-order effects from fJ
    b <- as.matrix(table(a1,a2))  #K1xK2 cell count matrix (rows correspond to X[,J[1]]; columns to X[,J[2]])
    b1 <- apply(b,1,sum)  #K1x1 count vector summed across X[,J[2]], as function of X[,J[1]]
    b2 <- apply(b,2,sum)  #K2x1 count vector summed across X[,J[1]], as function of X[,J[2]]
    Delta <- fJ[2:(K1+1),,,drop=FALSE]-fJ[1:K1,,,drop=FALSE] #K1x(K2+1) matrix of differenced fJ values, differenced across X[,J[1]]
    tmp <- (Delta[,1:K2,,drop=FALSE]+Delta[,2:(K2+1),,drop=FALSE])/2
    b.Delta <- array(b,c(K1,K2,length(tau)))*tmp
    Delta.Ave <- apply(b.Delta,c(1,3),sum)/b1
    fJ1 <- rbind(0,apply(Delta.Ave,2,cumsum))
    Delta <- fJ[,2:(K2+1),,drop=FALSE]-fJ[,1:K2,,drop=FALSE] #(K1+1)xK2 matrix of differenced fJ values, differenced across X[,J[2]]
    tmp <- (Delta[1:K1,,,drop=FALSE]+Delta[2:(K1+1),,,drop=FALSE])/2
    b.Delta <- array(b,c(K1,K2,length(tau)))*tmp
    Delta.Ave <- apply(b.Delta,c(2,3),sum)/b2
    fJ2 <- rbind(0,apply(Delta.Ave,2,cumsum))
    for (dd in 1:length(tau)) {
      fJ[,,dd] <- fJ[,,dd] - outer(fJ1[,dd],rep(1,K2+1)) - outer(rep(1,K1+1),fJ2[,dd])
    }
    x <- list(z1, z2)
    dimnames(fJ)[[3]] <- paste0(tau*100, "%")
    names(dimnames(fJ)) <- c(paste0("X",var.index),"tau")
    #end of "if (length(J) == 2)" statement
  }
  return(list(x = x, ALE = fJ))
}