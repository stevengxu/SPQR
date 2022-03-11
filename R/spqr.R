library(rlang)
library(Rcpp)
library(loo)
library(progressr)
library(matrixStats)
library(torch)
library(splines2)
library(yaImpute)
library(abind)
source("utils.R")
source("spqr.mcmc.R")
source("spqr.adam.R")


spqr.train <- function(params = list(), X, y, seed = NULL, verbose = TRUE, ...) {
  merged <- check.spqr.params(params, ...)
  if (merged[["method"]] == "MLE") {
    model <- spqr.adam.train(params=merged, X=X, y=y, seed=seed, verbose=verbose)
    class(model) <- c("spqr","adam")
  } else {
    model <- spqr.mcmc.train(params=merged, X=X, y=y, seed=seed, verbose=verbose)
    class(model) <- c("spqr","mcmc")
  }
  return(model)
}

spqr.predict <- function(object, X, y, result="qf", tau=seq(0.05,0.95,0.05)) {
  
  if (class(object)[2] == "mcmc") {
    Xt <- t(X)
    coefs <- 0
    nnn <- length(object$sample)
    for (i in 1:nnn) {
      W <- object$sample[[i]]$W
      b <- object$sample[[i]]$b
      coefs <- coefs + sp.coefs(W=W, b=b, X=Xt, activation=object$activation)/nnn
    }
    coefs <- t(coefs)
  } else {
    if (is.null(dim(X)))
      X <- as.matrix(X)
    
    X <- torch_tensor(X)
    model <- object$model
    model$eval()
    coefs <- as.matrix(model(X))
  }
  B <- sp.basis(y, K = object$n.knots, integral = (result != "pdf"))
  df <- coefs%*%B
  if (result != "qf") return(df)
  
  qf <- matrix(nrow = nrow(X), ncol = length(tau))
  colnames(qf) <- paste0(tau*100, "%")
  for(i in 1:nrow(qf)) qf[i,] <- approx(df[i,], y, xout = tau)$y 
  if (anyNA(qf))
    warning('Some extreme quantiles could not be calculated. ', 
            'Please increase the range of "y".')
  return(qf)
}

spqr.ale <-
  function(object, X, tau, J, K = 40, center = TRUE, y.grid = seq(0,1,0.01),
           pred.fun = NULL) {
    
  N <- dim(X)[1]  #sample size
  d <- dim(X)[2]  #number of predictor variables
  
  if (length(J) == 1) { #calculate main effects ALE plot
    
    if (class(X[,J]) == "numeric" | class(X[,J]) == "integer") { #for numerical or integer X[,J], calculate the ALE plot
      
      #find the vector of z values corresponding to the quantiles of X[,J]
      z <- c(min(X[,J]), as.numeric(quantile(X[,J],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values
      z <- unique(z)  #necessary if X[,J] is discrete, in which case z could have repeated values 
      K <- length(z)-1 #reset K to the number of unique quantile points
      fJ <- numeric(K)
      #group training rows into bins based on z
      a1 <- as.numeric(cut(X[,J], breaks=z, include.lowest=TRUE)) #N-length index vector indicating into which z-bin the training rows fall
      X1 <- X
      X2 <- X
      X1[,J] <- z[a1]
      X2[,J] <- z[a1+1]
      if (is.null(pred.fun)) {
        y.hat1 <- spqr.predict(object=object, X=X1, y=y.grid, result="qf", tau=tau)
        y.hat2 <- spqr.predict(object=object, X=X2, y=y.grid, result="qf", tau=tau)
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
      if (center) {
        b1 <- as.numeric(table(a1)) #frequency count of X[,J] values falling into z intervals
        if (K==1) {
          fJ <- t(t(fJ) - ((fJ[1:K,]+fJ[2:(K+1),])/2*b1)/sum(b1))
        } else {
          fJ <- t(t(fJ) - colSums((fJ[1:K,]+fJ[2:(K+1),])/2*b1)/sum(b1))  
        }
      }
      x <- z
      #plot(x, fJ, type="l", xlab=paste("x_",J, " (", names(X)[J], ")", sep=""), ylab= paste("f_",J,"(x_",J,")", sep=""))
    }  #end of else if (class(X[,J]) == "numeric" | class(X[,J]) == "integer") statement
    
    else print("error:  class(X[,J]) must be either numeric or integer")
    #end of if (length(J) == 1) statement  
  } else if (length(J) == 2) { #calculate second-order effects ALE plot
    
    if (class(X[,J[2]]) != "numeric" & class(X[,J[2]]) != "integer") {
      print("error:  Both X[,J[1]] and X[,J[2]] must be numeric or integer.")
    }
    
    #find the vectors of z values corresponding to the quantiles of X[,J[1]] and X[,J[2]]
    z1 <- c(min(X[,J[1]]), as.numeric(quantile(X[,J[1]],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values for X[,J[1]]
    z1 <- unique(z1)  #necessary if X[,J(1)] is discrete, in which case z1 could have repeated values 
    K1 <- length(z1)-1 #reset K1 to the number of unique quantile points
    #group training rows into bins based on z1
    a1 <- as.numeric(cut(X[,J[1]], breaks=z1, include.lowest=TRUE)) #N-length index vector indicating into which z1-bin the training rows fall
    z2 <- c(min(X[,J[2]]), as.numeric(quantile(X[,J[2]],seq(1/K,1,length.out=K), type=1)))  #vector of K+1 z values for X[,J[2]]
    z2 <- unique(z2)  #necessary if X[,J(2)] is discrete, in which case z2 could have repeated values 
    K2 <- length(z2)-1 #reset K2 to the number of unique quantile points
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
      y.hat11 <- spqr.predict(object=object, X=X11, y=y.grid, result="qf", tau=tau)
      y.hat12 <- spqr.predict(object=object, X=X12, y=y.grid, result="qf", tau=tau)
      y.hat21 <- spqr.predict(object=object, X=X21, y=y.grid, result="qf", tau=tau)
      y.hat22 <- spqr.predict(object=object, X=X22, y=y.grid, result="qf", tau=tau)
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
      if (K2==1) {
        fJ[,,dd] <- cumsum(Delta[,,dd])
      } else {
        fJ[,,dd] <- apply(t(apply(Delta[,,dd],1,cumsum)),2,cumsum)  
      }
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
    if (K2==1) {
      b.Delta <- matrix(b,nrow=K1,ncol=length(tau))*(Delta[,1:K2,]+Delta[,2:(K2+1),])/2
      Delta.Ave <- b.Delta/b1
      fJ1 <- rbind(0,apply(Delta.Ave,2,cumsum))
    } else {
      b.Delta <- array(b,c(K1,K2,length(tau)))*(Delta[,1:K2,]+Delta[,2:(K2+1),])/2
      Delta.Ave <- apply(b.Delta,c(1,3),sum)/b1
      fJ1 <- rbind(0,apply(Delta.Ave,2,cumsum))
    }
    Delta <- fJ[,2:(K2+1),]-fJ[,1:K2,] #(K1+1)xK2 matrix of differenced fJ values, differenced across X[,J[2]]
    if (K2==1) {
      b.Delta <- matrix(b,nrow=K1,ncol=length(tau))*(Delta[1:K1,]+Delta[2:(K1+1),])/2
      Delta.Ave <- colSums(b.Delta)/b2
      fJ2 <- rbind(0,Delta.Ave)
    } else {
      b.Delta <- array(b,c(K1,K2,length(tau)))*(Delta[1:K1,,]+Delta[2:(K1+1),,])/2
      Delta.Ave <- apply(b.Delta,c(2,3),sum)/b2
      fJ2 <- rbind(0,apply(Delta.Ave,2,cumsum))
    }
    for (dd in 1:length(tau)) {
      fJ[,,dd] <- fJ[,,dd] - outer(fJ1[,dd],rep(1,K2+1)) - outer(rep(1,K1+1),fJ2[,dd])
    }
    if (center) {
      if (K2==1) {
        fJ0 <- colSums(matrix(b,nrow=K1,ncol=length(tau))*(fJ[1:K1,1:K2,] + fJ[1:K1,2:(K2+1),] + fJ[2:(K1+1),1:K2,] + fJ[2:(K1+1), 2:(K2+1),])/4)/sum(b)
      } else {
        fJ0 <- apply(array(b,c(K1,K2,length(tau)))*(fJ[1:K1,1:K2,] + fJ[1:K1,2:(K2+1),] + fJ[2:(K1+1),1:K2,] + fJ[2:(K1+1), 2:(K2+1),])/4,3,sum)/sum(b)
      }
      for (dd in 1:length(tau)) {
        fJ[,,dd] <- fJ[,,dd] - fJ0[dd]
      }
    }
    x <- list(z1, z2)
    K <- c(K1, K2)
    #end of "if (length(J) == 2)" statement    
  } else { 
    print("error:  J must be a vector of length one or two")
  }
  out <- list(x.values=x, f.values = fJ)
  class(out) <- c("spqr","ale")
  return(out)
}


autoplot.spqr <- function(object, type = "ALE", index = NULL, tau = 0.5) {
  if(is.null(index))
    index <- NULL
}