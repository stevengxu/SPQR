source("spqr.R")

# Function to generate simulated data

rYgivenX <- function(n,p){
  X     <- matrix(runif(n*p),n,p)
  X[,1] <- 1
  m     <- 1/(1+exp(-1+5*X[,2]*X[,3]))
  Y     <- rbeta(n,10*m,10*(1-m))
  out   <- list(Y=Y,X=X)
  return(out)}

# PDF for the simulation design

dYgivenX <- function(y,X){
  m   <- 1/(1+exp(-1+5*X[2]*X[3]))
  out <- dbeta(y,10*m,10*(1-m))
  return(out)}

# QF for the simulation design

qYgivenX <- function(tau,X){
  m   <- 1/(1+exp(-1+5*X[2]*X[3]))
  out <- qbeta(tau,10*m,10*(1-m))
  return(out)}

set.seed(919)
n   <- 1000   # Number of observations
p   <- 5      # Number of covariates
dat <- rYgivenX(n,p)
X   <- dat$X
Y   <- dat$Y

n.knots <- 12
n.hidden <- 10 # can be a vector for deeper networks
activation <- "tanh" # "relu" also available


# not all parameters need to be specified, see utils.R for default values
adam.params <- list(
  lr = 0.01, # learning rate
  dropout = c(0,0), # a tuple with first element indicating dropout prob for input layer
                    # and second element for common dropout prob for hidden layers
  batchnorm = F,
  batch.size = 256,
  epochs = 500,
  patience = 10, # early stopping number
  model = NULL, # user supported torch model
  save.path = file.path(getwd(),"spqr_model"), # path to save the best torch model during validation
  save.name = "spqr.model.pt" # file to save the ...
)

mle.fit <- spqr.train(method="MLE", params=adam.params, X=X, y=Y, 
                      n.hidden=n.hidden, n.knots=n.knots, activation=activation)

mcmc.params <- list(
  sampler = "NUTS", # "HMC" also available
  prior = "ARD", # "ISO" and "GSM" also available
  control = list(adapt_delta = 0.9), # HMC control parameters
  iter = 1000, # total number of iterations
  warmup = 250, # warmup iters for stepsize and mass matrix adaptation
  thin = 5 # period for saving posterior samples
)

bayes.fit <- spqr.train(method="Bayes", params=mcmc.params, X=X, y=Y, 
                        n.hidden=n.hidden, n.knots=n.knots,
                        activation=activation, verbose=2)

X_test <- X[1:9,]
yyy <- seq(0,1,length.out=101)

pdf.mle <- spqr.predict(mle.fit, X_test, yyy, result = "pdf")
pdf.bayes <- spqr.predict(bayes.fit, X_test, yyy, result = "pdf")

par(mfrow=c(3,3))
for(i in 1:9){
  pdf0 <- dYgivenX(yyy,X_test[i,]) # True
  pdf1 <- pdf.mle[i,]     # MLE
  pdf2 <- pdf.bayes[i,]   # Bayes
  plot(yyy,pdf0,ylim=c(0,1.5*max(pdf0)),type="l",
       xlab="y",ylab="PDF",main=paste("Observation",i))
  lines(yyy,pdf1,col=2)
  lines(yyy,pdf2,col=3)
  if(i==1){
    legend("topright",c("True","MLE","Bayes"),lty=1,col=1:3,bty="n")
  }
}

tau <- seq(0.05,0.95,0.05)
qf.mle <- spqr.predict(mle.fit, X_test, yyy, result = "qf", tau=tau)
qf.bayes <- spqr.predict(bayes.fit, X_test, yyy, result = "qf", tau=tau)
par(mfrow=c(3,3))
for(i in 1:9){
  qf0 <- qYgivenX(tau,X_test[i,]) # True
  qf1 <- qf.mle[i,]     # MLE
  qf2 <- qf.bayes[i,]   # Bayes
  plot(tau,qf0,ylim=c(0,1.5*max(qf0)),type="l",
       xlab="tau",ylab="Quantile",main=paste("Observation",i))
  lines(tau,qf1,col=2)
  lines(tau,qf2,col=3)
  if(i==1){
    legend("topright",c("True","MLE","Bayes"),lty=1,col=1:3,bty="n")
  }
}

# Goodness-of-fit
cdf1 <- cdf2 <- {}
for (i in 1:length(Y)) {
  cdf1[i] <- spqr.predict(mle.fit, t(X[i,]), Y[i], result = "cdf")
  cdf2[i] <- spqr.predict(bayes.fit, t(X[i,]), Y[i], result = "cdf")
}
par(mfrow=c(1,2))
qqplot(cdf1, runif(n),xlab="CDF",ylab="U(0,1)",main="MLE")
abline(0,1,col=2,lwd=2)
qqplot(cdf2, runif(n),xlab="CDF",ylab="U(0,1)",main="Bayes")
abline(0,1,col=2,lwd=2)

# Sensitivity analysis
tau <- c(0.25,0.5,0.75)
pred.fun <- function(X, tau) {
  out <- matrix(nrow=nrow(X), ncol=length(tau))
  for (i in 1:nrow(X)) {
    out[i,] <- qYgivenX(tau, X[i,])
  }
  return(out)
}

# Main effect for x2, x3, x4
par(mfrow=c(3,3))
for (j in c(2,3,4)) {
  ale.mle <- spqr.ale(mle.fit, X, tau, J=j, center=F)
  ale.bayes <- spqr.ale(bayes.fit, X, tau, J=j, center=F)
  ale.ans <- spqr.ale(NULL, X, tau, J=j, center=F, pred.fun=pred.fun)
  
  for (i in 1:length(tau)) {
    plot(ale.ans$x.values, ale.ans$f.values[,i], type="l", xlab=paste0("x.",j), ylab="ALE", col=1)
    lines(ale.mle$x.values, ale.mle$f.values[,i], col=2)
    lines(ale.bayes$x.values, ale.bayes$f.values[,i], col=3)
    if(i==1){
      legend("topright",c("True","MLE","Bayes"),lty=1,col=1:3,bty="n")
    }
  }
}

# Interaction effect
par(mfrow=c(1,3))
ale.mle <- spqr.ale(mle.fit, X, tau, J=c(2,3), center=F)
ale.bayes <- spqr.ale(bayes.fit, X, tau, J=c(2,3), center=F)
ale.ans <- spqr.ale(NULL, X, tau, J=c(2,3), center=F, pred.fun=pred.fun)

image(ale.ans$f.values[,,1],xlab = "x.2",ylab = "x.3",main = "True")
image(ale.mle$f.values[,,1],xlab = "x.2",ylab = "x.3",main = "MLE")
image(ale.bayes$f.values[,,1],xlab = "x.2",ylab = "x.3",main = "Bayes")

par(mfrow=c(1,1))
