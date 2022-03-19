library(rlang)
library(Rcpp)
library(loo)
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
source("spqr.mcmc.R")
source("spqr.adam.R")


spqr.train <- function(params = list(), X, y, seed = NULL, verbose = TRUE, ...) {
  params <- check.spqr.params(params, ...)
  if (params[["method"]] == "Bayes") {
    model <- spqr.mcmc.train(params=params, X=X, y=y, seed=seed, verbose=verbose)
    class(model) <- c("spqr","mcmc")
  } else {
    model <- spqr.adam.train(params=params, X=X, y=y, seed=seed, verbose=verbose)
    class(model) <- c("spqr","adam")
  }
  invisible(model)
}

spqr.predict <- function(object, X, y, result = "qf", tau = seq(0.05,0.95,0.05), 
                         ci.level = 0.95) {
  if (is.null(dim(X)))
    dim(X) <- c(1,length(X))
  
  B <- sp.basis(y, K = object$n.knots, integral = (result != "pdf"))
  if (class(object)[2] == "mcmc") {
    nnn <- length(object$sample)
    if (is.null(ci.level)) {
      X <- t(X)
      coefs <- 0
      for (i in 1:nnn) {
        W <- object$sample[[i]]$W
        b <- object$sample[[i]]$b
        coefs <- coefs + sp.coefs(W=W, b=b, X=X, activation=object$activation)/nnn
      }
      df <- crossprod(coefs, B)
      if (result != "qf") 
        return(df)
      
      qf <- matrix(nrow = ncol(X), ncol = length(tau))
      colnames(qf) <- paste0(tau*100, "%")
      for(i in 1:nrow(qf)) 
        qf[i,] <- approx(df[i,], y, xout=tau, ties = list("ordered",min))$y 
      
      if (anyNA(qf))
        warning('Some extreme quantiles could not be calculated. ', 
                'Please increase the range of "y".')
      return(qf)
    } else {
      object.i <- list()
      object.i$n.knots <- object$n.knots
      object.i$activation <- object$activation
      class(object.i) <- class(object)
      dqf <- abind(lapply(seq_len(nnn), FUN = function(k) {
        object.i$sample <- list(object$sample[[k]])
        spqr.predict(object=object.i, X=X, y=y, result=result, tau=tau, ci.level=NULL)
      }), along=3)
      out <- array(dim=c(dim(dqf)[1:2],3))
      out[,,1] <- apply(dqf,1:2,quantile,probs=(1-ci.level)/2)
      out[,,2] <- apply(dqf,1:2,mean)
      out[,,3] <- apply(dqf,1:2,quantile,probs=(1+ci.level)/2)
      dimnames(out)[[3]] <- c("lower.bound","mean","upper.bound")
      return(out)
    }
  } else {
    X <- torch_tensor(X)
    model <- object$model
    model$eval()
    nn<- model(X)
    coefs <- as.matrix(nn$output)
    df <- coefs%*%B
    if (result != "qf") 
      return(df)
    
    qf <- matrix(nrow = nrow(X), ncol = length(tau))
    colnames(qf) <- paste0(tau*100, "%")
    for(i in 1:nrow(qf)) 
      qf[i,] <- approx(df[i,], y, xout=tau, ties = list("ordered",min))$y 
    
    if (anyNA(qf))
      warning('Some extreme quantiles could not be calculated. ', 
              'Please increase the range of "y".')
    return(qf)
  }
}

spqr.cv <- function(params = list(), X, y, nfold, folds=NULL, stratified=FALSE, 
                    verbose = TRUE, ...) {
  params <- check.spqr.params(params, ...)
  if (params[["method"]] == c("Bayes")) {
    stop("Cross-validation not available for `method` = 'Bayes'") 
  }
  if (!is.null(folds)) {
    if (!is.list(folds) || length(folds) < 2)
      stop("`folds` must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    nfold <- length(folds)
  } else {
    if (nfold <= 1)
      stop("`nfold` must be > 1")
    folds <- spqr.createFolds(y, nfold, stratified)
  }
  out <- spqr.adam.cv(params=params, X=X, y=y, folds=folds, verbose=verbose)
  invisible(out)
}

autoplot.spqr <- function(object, type = "ALE", var.index = NULL, tau = 0.5,
                          var.names = NULL, show.data = FALSE, ci.level = NULL, 
                          ...) {
  if (!is.numeric(var.index))
    stop("`var.index` must be a numeric vector.")
  if (!is.null(var.names) && length(var.names) != length(var.index))
    stop("`var.names` must have the same length as 'var.index'.")
  if (type == "ALE") {
    if (length(var.index) > 2)
      stop("For ALE plot, `var.index` must be a vector of length one or two.")
    ale <- spqr.ale(object, J=var.index, tau=tau, ci.level=ci.level, ...)
    if (length(var.index) == 1) {
      yrange <- range(ale$f.values)
      if (is.null(var.names)) {
        x.label <- parse(text=paste0("X[",var.index,"]"))
      } else {
        x.label <- var.names
      }
      df <- data.frame(x=ale$x.values)
      if (class(object)[2]=="mcmc" && !is.null(ci.level)) {
        figs <- lapply(seq_along(tau), FUN = function(i) {
          df$ymin <- ale$f.values[,i,"lower.bound"]
          df$y <- ale$f.values[,i,"mean"]
          df$ymax <- ale$f.values[,i,"upper.bound"]
          g <- ggplot(data=df, aes(x=x,y=y)) + geom_line(size=1.2) + 
            geom_ribbon(aes(x=x,ymin=ymin,ymax=ymax),colour="gray", alpha=0.5) +
            ylim(yrange) + 
            labs(x=x.label, y="ALE",title=parse(text=paste0("tau==",tau[i]))) + 
            theme_light() + 
            theme(axis.text = element_text(colour="black", size = 12),
                  axis.title=element_text(size=15),
                  plot.title=element_text(hjust=0.5, size=18, face="bold"))
          if (show.data) {
            xrug <- object$X[,var.index]
            g + geom_rug(data=data.frame(x=xrug), aes(x=x, y=NULL), alpha=0.3)
          } else {
            g
          }
        })
      } else {
        figs <- lapply(seq_along(tau), FUN = function(i) {
          df$y <- ale$f.values[,i]
          g <- ggplot(data=df, aes(x=x,y=y)) + geom_line(size=1.2) + 
            ylim(yrange) + 
            labs(x=x.label, y="ALE",title=parse(text=paste0("tau==",tau[i]))) + 
            theme_light() + 
            theme(axis.text = element_text(colour="black", size = 12),
                  axis.title=element_text(size=15),
                  plot.title=element_text(hjust=0.5, size=18, face="bold"))
          if (show.data) {
            xrug <- object$X[,var.index]
            g + geom_rug(data=data.frame(x=xrug), aes(x=x, y=NULL), alpha=0.3)
          } else {
            g
          }
        })
      }
    } else {
      mypalette <- colorRampPalette(rev(brewer.pal(10,"RdYlBu")))
      if (is.null(var.names)) {
        x1.label <- parse(text=paste0("X[",var.index[1],"]"))
        x2.label <- parse(text=paste0("X[",var.index[2],"]"))
      } else {
        x1.label <- var.names[1]
        x2.label <- var.names[2]
      }
      df <- expand.grid(x1=ale$x.values[[1]],x2=ale$x.values[[2]])
      figs <- lapply(seq_along(tau), FUN = function(i) {
        df$y <- c(ale$f.values[,,i])
        fld <- with(df, interp(x=x1, y=x2, z=y))
        df <- expand.grid(x1=fld$x,x2=fld$y)
        df$y <- c(fld$z)
        g <- ggplot(df,aes(x=x1, y=x2)) + 
          geom_raster(aes(fill=y)) + 
          geom_contour(aes(z=y), colour="black", size=1, alpha=0.5) + 
          geom_text_contour(aes(z=y), skip=0, colour="black", stroke=0.2, size=5) +
          labs(x=x1.label, y=x2.label, title=parse(text=paste0("tau==", tau[i]))) + 
          guides(fill="none") + 
          theme_light() + 
          theme(axis.text = element_text(colour = "black", size = 12),
                axis.title=element_text(size=15),
                plot.title=element_text(hjust=0.5, size=18, face="bold")) + 
          scale_fill_gradientn(colours=mypalette(20))
        if (show.data) {
          x1rug <- object$X[,var.index[1]]
          x2rug <- object$X[,var.index[2]]
          g + geom_rug(data=data.frame(x1=x1rug, x2=x2rug), aes(x=x1, y=x2), alpha=0.3)
        } else {
          g + scale_y_continuous(expand = c(0,0)) + scale_x_continuous(expand = c(0,0))
        }
      })
    }
  } else {
    if (length(var.index) == 1)
      stop("For VI plot, `var.index` must be a vector of length > 1.")
    x.ticks <- {}
    if (!is.null(var.names)) {
      x.ticks <- var.names
    } else {
      x.ticks <- parse(text=paste0("X[",var.index,"]"))
    }
    if (is.null(ci.level)) {
      vi <- matrix(nrow=length(var.index),ncol=length(tau))
      for (i in 1:nrow(vi)) {
        ale <- spqr.ale(object, J=var.index[i], tau=tau, ...)
        if (length(ale$x.values)>5) {
          vi[i,] <- colSds(ale$f.values)
        } else {
          vi[i,] <- (colMaxs(ale$f.values)-colMins(ale$f.values))/4
        }
      }
      df <- data.frame(x=var.index)
      figs <- lapply(seq_along(tau), FUN = function(i) {
        df$y <- vi[,i]
        x.ticks <- x.ticks[order(df$y, decreasing = T)]
        df <- df[order(df$y, decreasing = T),]
        ggplot(data=df, aes(x=reorder(x, y), y=y)) +
          geom_bar(stat = "identity",fill="#999999") +
          coord_flip() +
          labs(x=NULL, y=NULL, title=parse(text=paste0("tau==", tau[i]))) +
          theme_light() +
          theme(axis.title = element_text(size = 15, face="bold"),
                plot.title = element_text(hjust = 0.5,size = 18),
                axis.text.y = element_text(size = 12),
                axis.text.x = element_text(size = 12)) + 
          scale_x_discrete(labels=rev(x.ticks))
      }) 
    } else {
      nnn <- length(object$sample)
      out <- array(dim=c(length(var.index),length(tau),nnn))
      vi <- array(dim=c(length(var.index),length(tau),3))
      object.i <- list()
      object.i$n.knots <- object$n.knots
      object.i$activation <- object$activation
      class(object.i) <- class(object)
      for (i in 1:nrow(vi)) {
        out[i,,] <- abind(lapply(seq_len(length(object$sample)), FUN = function(k) {
          object.i$sample <- list(object$sample[[k]])
          ale <- .ale.main.sample(object=object.i, X=object$X, J=var.index[i], tau=tau, ...)
          if (length(unique(X[,var.index[i]]))>5) {
            colSds(ale)
          } else {
            (colMaxs(ale)-colMins(ale))/4
          }
        }), along=2)
        vi[i,,1] <- apply(out[i,,],1,quantile,probs=(1-ci.level)/2)
        vi[i,,2] <- apply(out[i,,],1,mean)
        vi[i,,3] <- apply(out[i,,],1,quantile,probs=(1+ci.level)/2)
      }
      dimnames(vi)[[3]] <- c("lower.bound","mean","upper.bound")
      df <- data.frame(x=var.index)
      figs <- lapply(seq_along(tau), FUN = function(i) {
        df$ymin <- vi[,i,"lower.bound"]
        df$y <- vi[,i,"mean"]
        df$ymax <- vi[,i,"upper.bound"]
        x.ticks <- x.ticks[order(df$y, decreasing = T)]
        df <- df[order(df$y, decreasing = T),]
        ggplot(data=df, aes(x=reorder(x, y), y=y)) +
          geom_bar(stat = "identity",fill="#999999") +
          coord_flip() +
          geom_errorbar(aes(ymin=ymin,ymax=ymax),color="#000000") + 
          labs(x=NULL, y=NULL, title=parse(text=paste0("tau==", tau[i]))) +
          theme_light() +
          theme(axis.title = element_text(size = 15, face="bold"),
                plot.title = element_text(hjust = 0.5,size = 18),
                axis.text.y = element_text(size = 12),
                axis.text.x = element_text(size = 12)) + 
          scale_x_discrete(labels=rev(x.ticks))
      }) 
    }
  }
  aa <- sqrt(length(tau))
  grid.row <- round(aa)
  grid.arrange(grobs=figs, nrow=grid.row)
}

spqr.ale <-
  function(object, J, tau, K = 40, y.grid = seq(0,1,0.01), pred.fun = NULL, 
           ci.level = NULL) {
    
  X <- object$X
  N <- dim(X)[1]  # sample size
  d <- dim(X)[2]  # number of predictor variables
  #center <- FALSE # centering is not meaningful for quantile predictions
  
  if (length(J) > 2)
    stop("`J` must be a vector of length one or two.")
  
  firstCheck <- class(X[,J[1]]) == "numeric" || class(X[,J[1]]) == "integer"
  if (length(J) == 1) { #calculate main effects ALE plot
    if (!firstCheck)
      stop("X[,J] must be numeric or integer.")
    #find the vector of z values corresponding to the quantiles of X[,J]
    z <- c(min(X[,J]), as.numeric(quantile(X[,J],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values
    z <- unique(z)  #necessary if X[,J] is discrete, in which case z could have repeated values 
    if (is.null(ci.level)) {
      K <- length(z)-1 #reset K to the number of unique quantile points
      fJ <- numeric(K)
      #group training rows into bins based on z
      a1 <- as.numeric(cut(X[,J], breaks=z, include.lowest=TRUE)) #N-length index vector indicating into which z-bin the training rows fall
      X1 <- X
      X2 <- X
      X1[,J] <- z[a1]
      X2[,J] <- z[a1+1]
      if (is.null(pred.fun)) {
        y.hat1 <- spqr.predict(object=object, X=X1, y=y.grid, result="qf", tau=tau, ci.level=NULL)
        y.hat2 <- spqr.predict(object=object, X=X2, y=y.grid, result="qf", tau=tau, ci.level=NULL)
      } else {
        y.hat1 <- pred.fun(X=X1, tau=tau)
        y.hat2 <- pred.fun(X=X2, tau=tau)
      }
      Delta <- y.hat2-y.hat1  #N-length vector of individual local effect values
      DDelta <- matrix(0,nrow = K, ncol = length(tau))
      for (i in 1:length(tau)) {
        DDelta[,i] <- as.numeric(tapply(Delta[,i], a1, mean)) #K-length vector of averaged local effect values
      }
      fJ <- rbind(0,apply(DDelta,2,cumsum)) #K+1 length vector
      
      #now vertically translate fJ, by subtracting its average (averaged across X[,J])
      #if (center) {
      #  b1 <- as.numeric(table(a1)) #frequency count of X[,J] values falling into z intervals
      #  if (K==1) {
      #    fJ <- t(t(fJ) - ((fJ[1:K,]+fJ[2:(K+1),])/2*b1)/sum(b1))
      #  } else {
      #    fJ <- t(t(fJ) - colSums((fJ[1:K,]+fJ[2:(K+1),])/2*b1)/sum(b1))  
      #  }
      #}
    } else {
      object.i <- list()
      object.i$n.knots <- object$n.knots
      object.i$activation <- object$activation
      class(object.i) <- class(object)
      out <- abind(lapply(seq_len(length(object$sample)), FUN = function(k) {
        object.i$sample <- list(object$sample[[k]])
        .ale.main.sample(object=object.i, X=X, J=J, tau=tau, K=K, y.grid=y.grid)
      }), along=3)
      fJ <- array(0,dim=c(K+1, length(tau), 3))
      fJ[,,1] <- apply(out,1:2,quantile,probs=(1-ci.level)/2)
      fJ[,,2] <- apply(out,1:2,mean)
      fJ[,,3] <- apply(out,1:2,quantile,probs=(1+ci.level)/2)
      dimnames(fJ)[[3]] <- c("lower.bound","mean","upper.bound")
    }
    x <- z
    colnames(fJ) <- paste0(tau*100, "%") 
  #end of if (length(J) == 1) statement  
  } else { #calculate second-order effects ALE plot
    secondCheck <- class(X[,J[2]]) == "numeric" || class(X[,J[2]]) == "integer"
    if (!(firstCheck && secondCheck))
      stop("Both X[,J[1]] and X[,J[2]] must be numeric or integer.")
    
    #find the vectors of z values corresponding to the quantiles of X[,J[1]] and X[,J[2]]
    z1 <- c(min(X[,J[1]]), as.numeric(quantile(X[,J[1]],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values for X[,J[1]]
    z1 <- unique(z1)  #necessary if X[,J(1)] is discrete, in which case z1 could have repeated values 
    K1 <- length(z1)-1 #reset K1 to the number of unique quantile points
    if (K1 == 1)
      stop("X[,J[1]] should have at least 3 unique values.")
    #group training rows into bins based on z1
    a1 <- as.numeric(cut(X[,J[1]], breaks=z1, include.lowest=TRUE)) #N-length index vector indicating into which z1-bin the training rows fall
    z2 <- c(min(X[,J[2]]), as.numeric(quantile(X[,J[2]],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values for X[,J[2]]
    z2 <- unique(z2)  #necessary if X[,J(2)] is discrete, in which case z2 could have repeated values 
    K2 <- length(z2)-1 #reset K2 to the number of unique quantile points
    if (K2 == 1)
      stop("X[,J[2]] should have at least 3 unique values.")
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
      y.hat11 <- spqr.predict(object=object, X=X11, y=y.grid, result="qf", tau=tau, ci.level=NULL)
      y.hat12 <- spqr.predict(object=object, X=X12, y=y.grid, result="qf", tau=tau, ci.level=NULL)
      y.hat21 <- spqr.predict(object=object, X=X21, y=y.grid, result="qf", tau=tau, ci.level=NULL)
      y.hat22 <- spqr.predict(object=object, X=X22, y=y.grid, result="qf", tau=tau, ci.level=NULL)
    } else {
      y.hat11 <- pred.fun(X=X11, tau=tau)
      y.hat12 <- pred.fun(X=X12, tau=tau)
      y.hat21 <- pred.fun(X=X21, tau=tau)
      y.hat22 <- pred.fun(X=X22, tau=tau)
    }
    Delta <- (y.hat22-y.hat21)-(y.hat12-y.hat11)  #N-length vector of individual local effect values
    Delta <- abind(lapply(seq_len(ncol(Delta)),function(dd) {
      as.matrix(tapply(Delta[,dd], list(a1, a2), mean))
    }),along = 3)#K1xK2 matrix of averaged local effects, which includes NA values if a cell is empty
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
        }#Set Delta for NA cells equal to Delta for their closest neighbor non-NA cell. The matrix() command is needed, because if there is only one empty cell, notNA.ind[nbrs] is created as a 2-length vector instead of a 1x2 matrix, which does not index Delta properly 
      } #end of if (nrow(NA.ind) > 0) statement
      #accumulate the values in Delta
    }
    fJ <- array(0,c(K1,K2,length(tau)))
    for (dd in 1:length(tau)) {
      fJ[,,dd] <- apply(t(apply(Delta[,,dd],1,cumsum)),2,cumsum)  
    }
    fJ_ <- fJ
    fJ <- array(0,c(K1+1,K2+1,length(tau)))
    fJ[-1,-1,] <- fJ_
    #add a first row and first column to fJ that are all zeros
    #now subtract the lower-order effects from fJ
    b <- as.matrix(table(a1,a2))  #K1xK2 cell count matrix (rows correspond to X[,J[1]]; columns to X[,J[2]])
    b1 <- apply(b,1,sum)  #K1x1 count vector summed across X[,J[2]], as function of X[,J[1]]
    b2 <- apply(b,2,sum)  #K2x1 count vector summed across X[,J[1]], as function of X[,J[2]]
    Delta <- fJ[2:(K1+1),,]-fJ[1:K1,,] #K1x(K2+1) matrix of differenced fJ values, differenced across X[,J[1]]
    if (length(tau)==1) {
      tmp <- (Delta[,1:K2]+Delta[,2:(K2+1)])/2
      dim(tmp) <- c(dim(tmp),1)
    } else {
      tmp <- (Delta[,1:K2,]+Delta[,2:(K2+1),])/2
    }
    b.Delta <- array(b,c(K1,K2,length(tau)))*tmp
    Delta.Ave <- apply(b.Delta,c(1,3),sum)/b1
    fJ1 <- rbind(0,apply(Delta.Ave,2,cumsum))
    Delta <- fJ[,2:(K2+1),]-fJ[,1:K2,] #(K1+1)xK2 matrix of differenced fJ values, differenced across X[,J[2]]
    if (length(tau)==1){
      tmp <- (Delta[1:K1,]+Delta[2:(K1+1),])/2
      dim(tmp) <- c(dim(tmp),1)
    } else {
      tmp <- (Delta[1:K1,,]+Delta[2:(K1+1),,])/2
    }
    b.Delta <- array(b,c(K1,K2,length(tau)))*tmp
    Delta.Ave <- apply(b.Delta,c(2,3),sum)/b2
    fJ2 <- rbind(0,apply(Delta.Ave,2,cumsum))
    for (dd in 1:length(tau)) {
      fJ[,,dd] <- fJ[,,dd] - outer(fJ1[,dd],rep(1,K2+1)) - outer(rep(1,K1+1),fJ2[,dd])
    }
    #if (center) {
    #  fJ0 <- apply(array(b,c(K1,K2,length(tau)))*(fJ[1:K1,1:K2,] + fJ[1:K1,2:(K2+1),] + fJ[2:(K1+1),1:K2,] + fJ[2:(K1+1), 2:(K2+1),])/4,3,sum)/sum(b)
    #  for (dd in 1:length(tau)) {
    #    fJ[,,dd] <- fJ[,,dd] - fJ0[dd]
    #  }
    #}
    x <- list(z1, z2)
    K <- c(K1, K2)
    dimnames(fJ)[[3]] <- paste0(tau*100, "%")
    #end of "if (length(J) == 2)" statement    
  }
  out <- list(x.values=x, f.values = fJ)
  class(out) <- c("spqr","ale")
  return(out)
}


.ale.main.sample <- function(object, X, J, tau, K = 40, y.grid = seq(0,1,0.01)) {
  # Helper function for calculating credible intervals of ALEs
  # Not meant to be used directly
  
  N <- dim(X)[1]
  d <- dim(X)[2]
  z <- c(min(X[,J]), as.numeric(quantile(X[,J],seq(1/K,1,length.out=K), type=1)))
  z <- unique(z)
  K <- length(z)-1 
  fJ <- numeric(K)

  a1 <- as.numeric(cut(X[,J], breaks=z, include.lowest=TRUE)) 
  X1 <- X
  X2 <- X
  X1[,J] <- z[a1]
  X2[,J] <- z[a1+1]
  y.hat1 <- spqr.predict(object=object, X=X1, y=y.grid, result="qf", tau=tau, ci.level=NULL)
  y.hat2 <- spqr.predict(object=object, X=X2, y=y.grid, result="qf", tau=tau, ci.level=NULL)
  Delta <- y.hat2-y.hat1  
  DDelta <- matrix(0,nrow = K, ncol = length(tau))
  for (i in 1:length(tau)) {
    DDelta[,i] <- as.numeric(tapply(Delta[,i], a1, mean))
  }
  fJ <- rbind(0,apply(DDelta,2,cumsum))
  return(fJ)
}
